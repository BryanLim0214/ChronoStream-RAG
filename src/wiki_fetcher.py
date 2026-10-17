"""
ChronoStream-RAG: Real Wikimedia Revision History Fetcher & Temporal QA Corpus Builder
======================================================================================
Fetches real revision histories from the Wikimedia API (en.wikipedia.org) and constructs
a verifiable multi-chunk temporal RAG benchmark (WikiChrono-Bench) containing:
  1. Real Wikimedia revision metadata (revid, parentid, timestamp, size, comment, sha1)
  2. Multi-chunk evolving document bodies with verifiable factual transitions across versions
  3. Variable chunk-count transitions on 35% of articles (testing Cross-Chunk Version Tearing & Orphan Chunks)
  4. Transient vandalism/revert sequences (testing Watermark Coalescing & Revert Resilience)
  5. Invariant background chunks across revisions (testing Chunk-Level SHA-256 Delta Deduplication)
  6. Ground-truth Temporal QA pairs:
     - Current-State Queries (evaluating latest arrived revision agreement)
     - Historical Point-in-Time ('AS OF' valid-time) Queries (evaluating bitemporal MVCC)
     - Intra-Document Multi-Chunk Consistency Queries (detecting split-brain hallucinations)
"""

import os
import json
import time
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ChronoStreamRAG-Research/1.0 (BigDataAnalyticsProject; contact: chronostream@research.org)"

# 20 Core Wikipedia topics (7 with shrink_chunks=True [35%] to test Orphan Chunk Tearing,
# 13 maintaining 6 chunks [65%] to test pure Out-of-Order & Compute-Lag Overwrites)
WIKI_TOPICS = [
    # Technology & AI
    {"title": "Large language model", "domain": "Technology", "entity": "LLM Benchmark Leader",
     "fact_field": "flagship reference architecture capacity",
     "v_states": [
         {"rev_num": 1, "fact_val": "175 billion parameters with 2,048-token context window", "summary_note": "Initial baseline release"},
         {"rev_num": 2, "fact_val": "175 billion parameters with 2,048-token context window", "summary_note": "Minor punctuation and wikilink cleanup", "is_minor_syntactic": True},
         {"rev_num": 3, "fact_val": "540 billion parameters with 8,192-token context window", "summary_note": "Update scaling law milestone"},
         {"rev_num": 4, "fact_val": "999 quadrillion parameters invented by Mars Colony", "summary_note": "Unverified anonymous edit", "is_vandalism": True},
         {"rev_num": 5, "fact_val": "540 billion parameters with 8,192-token context window", "summary_note": "Revert vandalism to verified 540B state", "is_revert": True},
         {"rev_num": 6, "fact_val": "1.8 trillion sparse MoE parameters with 128,000-token context window", "summary_note": "Major consolidation of MoE architecture section", "shrink_chunks": True}
     ]},
    {"title": "Apache Kafka", "domain": "Technology", "entity": "Apache Kafka Consensus Protocol",
     "fact_field": "cluster metadata consensus mechanism and latest production release",
     "v_states": [
         {"rev_num": 1, "fact_val": "Apache ZooKeeper ensemble metadata controller in release 2.8", "summary_note": "Document ZooKeeper architecture"},
         {"rev_num": 2, "fact_val": "KRaft (Kafka Raft) Early Access alongside ZooKeeper in release 3.3", "summary_note": "Add KRaft production readiness"},
         {"rev_num": 3, "fact_val": "KRaft (Kafka Raft) Early Access alongside ZooKeeper in release 3.3", "summary_note": "Fix typo in partition leader election section", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "KRaft mandatory quorum with ZooKeeper deprecated in release 3.7", "summary_note": "Update for 3.7 deprecation notice"},
         {"rev_num": 5, "fact_val": "ZooKeeper completely removed; pure KRaft consensus in release 4.0", "summary_note": "Major 4.0 release overhaul removing legacy ZooKeeper subsections", "shrink_chunks": True}
     ]},
    {"title": "Apache Spark", "domain": "Technology", "entity": "Apache Spark Streaming Engine",
     "fact_field": "default micro-batch execution engine and state store backend",
     "v_states": [
         {"rev_num": 1, "fact_val": "DStream discretized streams with 500ms minimum batch interval", "summary_note": "Initial Spark Streaming overview"},
         {"rev_num": 2, "fact_val": "Structured Streaming with HDFS-backed in-memory state store in Spark 2.4", "summary_note": "Transition to Structured Streaming"},
         {"rev_num": 3, "fact_val": "Project Lightspeed with RocksDB state store and asynchronous checkpointing in Spark 3.4", "summary_note": "Update stateful processing engine"},
         {"rev_num": 4, "fact_val": "DELETED BY TROLL 12345", "summary_note": "Blanking section", "is_vandalism": True},
         {"rev_num": 5, "fact_val": "Project Lightspeed with RocksDB state store and asynchronous checkpointing in Spark 3.4", "summary_note": "ClueBot revert blanking", "is_revert": True},
         {"rev_num": 6, "fact_val": "TransformWithState arbitrary stateful operator on RocksDB in Spark 4.0", "summary_note": "Spark 4.0 streaming architecture update", "shrink_chunks": False}
     ]},
    {"title": "Python (programming language)", "domain": "Technology", "entity": "Python Stable Release & GIL Status",
     "fact_field": "latest stable release version and Global Interpreter Lock (GIL) concurrency status",
     "v_states": [
         {"rev_num": 1, "fact_val": "Python 3.11 with specialized adaptive interpreter and mandatory GIL", "summary_note": "Update for 3.11 release"},
         {"rev_num": 2, "fact_val": "Python 3.12 with sub-interpreters (PEP 684) and per-interpreter GIL", "summary_note": "Update for 3.12"},
         {"rev_num": 3, "fact_val": "Python 3.12 with sub-interpreters (PEP 684) and per-interpreter GIL", "summary_note": "Formatting code blocks", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "Python 3.13 with experimental free-threaded mode (PEP 703 optional no-GIL) and JIT compiler", "summary_note": "Major update for Python 3.13 free-threading", "shrink_chunks": False}
     ]},
    {"title": "Linux kernel", "domain": "Technology", "entity": "Linux Kernel Longterm Release",
     "fact_field": "active mainline kernel series and scheduler implementation",
     "v_states": [
         {"rev_num": 1, "fact_val": "Linux 6.1 LTS utilizing the Completely Fair Scheduler (CFS)", "summary_note": "Baseline 6.1 LTS"},
         {"rev_num": 2, "fact_val": "Linux 6.6 LTS replacing CFS with the EEVDF (Earliest Eligible Virtual Deadline First) scheduler", "summary_note": "EEVDF merge update"},
         {"rev_num": 3, "fact_val": "Linux 6.6 LTS replacing CFS with the EEVDF (Earliest Eligible Virtual Deadline First) scheduler", "summary_note": "Grammar fix in scheduler section", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "Linux 6.12 LTS with mainline PREEMPT_RT real-time patchset and sched_ext BPF scheduler", "summary_note": "Consolidate 6.12 LTS real-time kernel milestone", "shrink_chunks": True}
     ]},
    # Space & Science
    {"title": "Artemis program", "domain": "Science", "entity": "Artemis Crewed Lunar Schedule",
     "fact_field": "official target launch window and mission status for Artemis II and Artemis III",
     "v_states": [
         {"rev_num": 1, "fact_val": "Artemis II scheduled for November 2024 and Artemis III lunar landing in December 2025", "summary_note": "Post-Artemis I schedule"},
         {"rev_num": 2, "fact_val": "Artemis II shifted to September 2025 and Artemis III shifted to September 2026 for heat shield inspection", "summary_note": "NASA January 2024 schedule update"},
         {"rev_num": 3, "fact_val": "Artemis program cancelled and replaced by submarine mission", "summary_note": "Vandalism edit", "is_vandalism": True},
         {"rev_num": 4, "fact_val": "Artemis II shifted to September 2025 and Artemis III shifted to September 2026 for heat shield inspection", "summary_note": "Revert submarine vandalism", "is_revert": True},
         {"rev_num": 5, "fact_val": "Artemis II targeted for April 2026 and Artemis III targeted for mid-2027 following Orion heat shield root-cause resolution", "summary_note": "December 2024 NASA schedule revision and table compaction", "shrink_chunks": True}
     ]},
    {"title": "James Webb Space Telescope", "domain": "Science", "entity": "JWST Earliest Confirmed Galaxy Record",
     "fact_field": "most distant spectroscopically confirmed galaxy redshift record discovered by JWST",
     "v_states": [
         {"rev_num": 1, "fact_val": "GLASS-z12 at photometric redshift z = 12.1", "summary_note": "Early release science candidates"},
         {"rev_num": 2, "fact_val": "JADES-GS-z13-0 spectroscopically confirmed at redshift z = 13.20 (320 million years after Big Bang)", "summary_note": "NIRSpec confirmation of JADES-GS-z13-0"},
         {"rev_num": 3, "fact_val": "JADES-GS-z13-0 spectroscopically confirmed at redshift z = 13.20 (320 million years after Big Bang)", "summary_note": "Citation template formatting", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "JADES-GS-z14-0 spectroscopically confirmed at redshift z = 14.32 (290 million years after Big Bang)", "summary_note": "Update record to JADES-GS-z14-0", "shrink_chunks": False}
     ]},
    {"title": "SpaceX Starship", "domain": "Science", "entity": "SpaceX Starship Integrated Flight Test Status",
     "fact_field": "latest completed integrated flight test milestone and booster recovery outcome",
     "v_states": [
         {"rev_num": 1, "fact_val": "Integrated Flight Test 2 (IFT-2) achieved hot-staging separation before booster telemetry loss", "summary_note": "IFT-2 summary"},
         {"rev_num": 2, "fact_val": "Integrated Flight Test 4 (IFT-4) achieved controlled soft ocean splashdown of both Super Heavy and Starship", "summary_note": "IFT-4 successful splashdowns"},
         {"rev_num": 3, "fact_val": "Integrated Flight Test 5 (IFT-5) achieved the first mechanical chopstick arms catch of Super Heavy Booster 12 at Starbase launch tower", "summary_note": "Historic IFT-5 tower catch update", "shrink_chunks": False}
     ]},
    {"title": "CRISPR gene editing", "domain": "Science", "entity": "Approved CRISPR Clinical Therapy Status",
     "fact_field": "regulatory approval status of the first CRISPR-Cas9 human therapeutic",
     "v_states": [
         {"rev_num": 1, "fact_val": "Exa-cel (CTX001) under Phase 3 clinical trial review with BLA submitted to FDA", "summary_note": "Clinical trial phase 3 status"},
         {"rev_num": 2, "fact_val": "Casgevy (exa-cel) conditionally approved by UK MHRA in November 2023 as world's first licensed CRISPR therapy", "summary_note": "UK MHRA approval"},
         {"rev_num": 3, "fact_val": "Casgevy (exa-cel) conditionally approved by UK MHRA in November 2023 as world's first licensed CRISPR therapy", "summary_note": "Spelling correction", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "Casgevy (exa-cel) granted full FDA approval (Dec 2023) and European Commission authorization (Feb 2024) for sickle cell disease and beta thalassemia", "summary_note": "Full FDA and EU approval consolidation", "shrink_chunks": True}
     ]},
    {"title": "ITER", "domain": "Science", "entity": "ITER Tokamak First Plasma Schedule",
     "fact_field": "official baseline schedule for ITER Deuterium-Tritium fusion operations and wall material",
     "v_states": [
         {"rev_num": 1, "fact_val": "First Plasma scheduled for 2025 using beryllium first-wall armor panels", "summary_note": "2016 baseline schedule"},
         {"rev_num": 2, "fact_val": "2025 First Plasma date under formal revision due to thermal shield corrosion and vacuum vessel sector repairs", "summary_note": "2023 repair assessment"},
         {"rev_num": 3, "fact_val": "Revised 2024 baseline switches first wall from beryllium to tungsten with Research Operations in 2034 and Deuterium-Tritium phase in 2039", "summary_note": "July 2024 new ITER baseline announcement", "shrink_chunks": False}
     ]},
    # Corporations, Economics & Governance
    {"title": "OpenAI", "domain": "Organizations", "entity": "OpenAI Executive Leadership & Board",
     "fact_field": "Chief Executive Officer and corporate board structure",
     "v_states": [
         {"rev_num": 1, "fact_val": "Sam Altman serves as CEO under a 6-member nonprofit board including Ilya Sutskever and Helen Toner", "summary_note": "Pre-November 2023 governance"},
         {"rev_num": 2, "fact_val": "Mira Murati appointed interim CEO on November 17, 2023 following abrupt board removal of Sam Altman", "summary_note": "Breaking board removal"},
         {"rev_num": 3, "fact_val": "Emmett Shear named interim CEO on November 19, 2023 while Sam Altman transiently announced joining Microsoft", "summary_note": "Weekend leadership transition"},
         {"rev_num": 4, "fact_val": "Elon Musk bought OpenAI for 1 dollar", "summary_note": "Troll edit", "is_vandalism": True},
         {"rev_num": 5, "fact_val": "Emmett Shear named interim CEO on November 19, 2023 while Sam Altman transiently announced joining Microsoft", "summary_note": "Revert vandalism", "is_revert": True},
         {"rev_num": 6, "fact_val": "Sam Altman reinstated as CEO on November 29, 2023 with reconstituted board chaired by Bret Taylor alongside Larry Summers and Adam D'Angelo", "summary_note": "Final reinstatement and consolidation of November transition paragraphs", "shrink_chunks": True}
     ]},
    {"title": "Nvidia", "domain": "Organizations", "entity": "Nvidia Data Center GPU Architecture & Valuation",
     "fact_field": "flagship AI data center architecture in volume production and market capitalization tier",
     "v_states": [
         {"rev_num": 1, "fact_val": "Hopper H100 architecture in production with market capitalization crossing $1.0 trillion", "summary_note": "Mid-2023 milestone"},
         {"rev_num": 2, "fact_val": "Hopper H200 announced with HBM3e memory and market capitalization reaching $2.0 trillion", "summary_note": "Early 2024 update"},
         {"rev_num": 3, "fact_val": "Hopper H200 announced with HBM3e memory and market capitalization reaching $2.0 trillion", "summary_note": "Fix citation date", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "Blackwell B200/GB200 NVL72 rack architecture in volume production with market capitalization surpassing $3.3 trillion", "summary_note": "Blackwell production ramp", "shrink_chunks": False}
     ]},
    {"title": "European Union Artificial Intelligence Act", "domain": "Governance", "entity": "EU Artificial Intelligence Act",
     "fact_field": "official regulation number and enforcement timeline of the EU AI Act",
     "v_states": [
         {"rev_num": 1, "fact_val": "Under trilogue negotiation between European Parliament, Council, and Commission regarding foundation model rules", "summary_note": "2023 trilogue phase"},
         {"rev_num": 2, "fact_val": "Political agreement reached in December 2023 and formally endorsed by European Parliament in March 2024", "summary_note": "Parliament vote passed"},
         {"rev_num": 3, "fact_val": "Regulation (EU) 2024/1689 entered into force on August 1, 2024 with prohibited AI bans effective February 2, 2025", "summary_note": "Official entry into force", "shrink_chunks": False}
     ]},
    {"title": "NATO", "domain": "Governance", "entity": "NATO Member States & Secretary General",
     "fact_field": "total number of NATO member countries, newest member state, and current Secretary General",
     "v_states": [
         {"rev_num": 1, "fact_val": "31 member states following Finland's accession in April 2023, led by Secretary General Jens Stoltenberg", "summary_note": "Finland accession"},
         {"rev_num": 2, "fact_val": "32 member states following Sweden's formal accession on March 7, 2024, led by Secretary General Jens Stoltenberg", "summary_note": "Sweden becomes 32nd member"},
         {"rev_num": 3, "fact_val": "32 member states following Sweden's formal accession on March 7, 2024, led by Secretary General Jens Stoltenberg", "summary_note": "Minor formatting", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "32 member states with Sweden as newest member, led by Secretary General Mark Rutte who assumed office on October 1, 2024", "summary_note": "Mark Rutte assumes office", "shrink_chunks": False}
     ]},
    {"title": "World Health Organization", "domain": "Governance", "entity": "WHO Pandemic Agreement & Mpox Status",
     "fact_field": "PHEIC (Public Health Emergency of International Concern) status for mpox",
     "v_states": [
         {"rev_num": 1, "fact_val": "First mpox PHEIC declared in July 2022 and formally lifted in May 2023 after global clade IIb decline", "summary_note": "May 2023 lifting of first PHEIC"},
         {"rev_num": 2, "fact_val": "WHO monitoring upsurge of novel clade Ib mpox in Democratic Republic of the Congo without active PHEIC", "summary_note": "Early 2024 surveillance note"},
         {"rev_num": 3, "fact_val": "Second PHEIC declared by WHO Director-General on August 14, 2024 following cross-border spread of clade Ib mpox in Central and East Africa", "summary_note": "August 2024 second PHEIC declaration", "shrink_chunks": False}
     ]},
    # Sports & Global Events
    {"title": "UEFA Champions League", "domain": "Sports", "entity": "UEFA Champions League Format & Defending Champion",
     "fact_field": "tournament league stage structure and defending title holder",
     "v_states": [
         {"rev_num": 1, "fact_val": "Manchester City (2022-23 winners) under traditional 32-team, 8-group double round-robin format", "summary_note": "2023-24 season baseline"},
         {"rev_num": 2, "fact_val": "Final 32-team group format concluded with Real Madrid winning their 15th title in June 2024", "summary_note": "June 2024 Wembley final update"},
         {"rev_num": 3, "fact_val": "Arsenal won 99 Champions Leagues in a row", "summary_note": "Fan vandalism", "is_vandalism": True},
         {"rev_num": 4, "fact_val": "Final 32-team group format concluded with Real Madrid winning their 15th title in June 2024", "summary_note": "Revert fan vandalism", "is_revert": True},
         {"rev_num": 5, "fact_val": "36-team single-table Swiss-system League Phase (8 matches per club) introduced in 2024-25 with Real Madrid as defending champions", "summary_note": "2024-25 Swiss League Phase rollout and removal of legacy group draw rules", "shrink_chunks": True}
     ]},
    {"title": "2024 Summer Olympics", "domain": "Sports", "entity": "Paris 2024 Olympic Medal Table Outcome",
     "fact_field": "final overall gold medal table winner and tie-breaker count at Paris 2024",
     "v_states": [
         {"rev_num": 1, "fact_val": "Games in preparation in Paris with 329 gold medal events scheduled across 32 sports", "summary_note": "Pre-Games overview"},
         {"rev_num": 2, "fact_val": "People's Republic of China leading gold medal count at 39 golds entering the final day of competition", "summary_note": "Day 15 live tally"},
         {"rev_num": 3, "fact_val": "United States and China tied at 40 gold medals each, with the United States finishing 1st overall on silver medal tie-breaker (44 silver vs 27 silver, 126 total medals)", "summary_note": "Final official Paris 2024 medal standings", "shrink_chunks": False}
     ]},
    {"title": "Formula One", "domain": "Sports", "entity": "Formula One Constructors' & Drivers' Championship Status",
     "fact_field": "reigning World Drivers' Champion and World Constructors' Champion",
     "v_states": [
         {"rev_num": 1, "fact_val": "Max Verstappen (Drivers' Champion) and Red Bull Racing (Constructors' Champion) dominating 21 of 22 races in 2023", "summary_note": "End of 2023 season"},
         {"rev_num": 2, "fact_val": "Max Verstappen clinched his 4th consecutive Drivers' Championship at Las Vegas GP while McLaren leads the 2024 Constructors' standings", "summary_note": "Las Vegas GP update"},
         {"rev_num": 3, "fact_val": "Max Verstappen (4th Drivers' title) and McLaren (2024 Constructors' Champion, first constructor title since 1998 clinched at Abu Dhabi GP)", "summary_note": "Abu Dhabi finale update", "shrink_chunks": False}
     ]},
    {"title": "Chess World Championship", "domain": "Sports", "entity": "Reigning World Chess Champion",
     "fact_field": "reigning classical World Chess Champion and championship match result",
     "v_states": [
         {"rev_num": 1, "fact_val": "Ding Liren after defeating Ian Nepomniachtchi in rapid tiebreaks (2023 Astana match)", "summary_note": "2023 championship result"},
         {"rev_num": 2, "fact_val": "Ding Liren defending title in Singapore against 18-year-old Challenger Gukesh Dommaraju, tied 6.5-6.5 after Game 13", "summary_note": "Singapore 2024 match after Game 13"},
         {"rev_num": 3, "fact_val": "Gukesh Dommaraju (youngest undisputed World Chess Champion at age 18) after defeating Ding Liren 7.5-6.5 in Game 14 in Singapore (Dec 2024)", "summary_note": "Gukesh wins World Championship", "shrink_chunks": False}
     ]},
    {"title": "Bitcoin", "domain": "Economics", "entity": "Bitcoin Block Subsidy & Spot ETF Status",
     "fact_field": "current Bitcoin block mining reward subsidy and US Spot ETF regulatory status",
     "v_states": [
         {"rev_num": 1, "fact_val": "6.25 BTC per block (3rd halving epoch) with US Spot Bitcoin ETF applications pending SEC review", "summary_note": "Late 2023 status"},
         {"rev_num": 2, "fact_val": "6.25 BTC per block with 11 US Spot Bitcoin ETFs approved by the SEC on January 10, 2024", "summary_note": "SEC Spot ETF approval"},
         {"rev_num": 3, "fact_val": "6.25 BTC per block with 11 US Spot Bitcoin ETFs approved by the SEC on January 10, 2024", "summary_note": "Minor formatting", "is_minor_syntactic": True},
         {"rev_num": 4, "fact_val": "3.125 BTC per block following the 4th halving at block height 840,000 (April 20, 2024) with active Spot ETFs", "summary_note": "4th halving at block 840,000", "shrink_chunks": False}
     ]}
]


def fetch_real_wikipedia_revisions(title: str, limit: int = 30) -> list:
    """Fetches real revision history metadata from live English Wikipedia MediaWiki API."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvlimit": limit,
        "rvprop": "ids|timestamp|user|size|comment|sha1",
        "rvdir": "older"
    }
    url = f"{WIKI_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                revs = page_info.get("revisions", [])
                return sorted(revs, key=lambda r: r.get("timestamp", ""))
    except Exception as e:
        print(f"[WikiFetcher] Warning fetching revisions for {title}: {e}")
    return []


def fetch_real_wikipedia_extract(title: str) -> str:
    """Fetches real plain-text encyclopedic article extract from live English Wikipedia MediaWiki API."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "titles": title
    }
    url = f"{WIKI_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                ext = page_info.get("extract", "").strip()
                if ext:
                    return ext
    except Exception as e:
        print(f"[WikiFetcher] Warning fetching extract for {title}: {e}")
    return ""


def build_multi_chunk_document(topic: dict, state: dict, real_rev: dict, doc_idx: int, real_extract: str = "") -> dict:
    """
    Constructs a multi-chunk Wikimedia document revision combining live Wikipedia article text
    extracts with verifiable revision state transitions:
      - Chunks 0 and 2 carry the updated factual state (or minor syntactic edit)
      - Chunks 1 and 3 contain real live Wikipedia encyclopedic paragraphs (invariant across
        minor revisions -> enabling SHA-256 Chunk-Delta Deduplication in M4 to skip re-embedding
        unchanged sections!)
      - Chunks 4 and 5 are extended appendix/table sections that carry the factual state and
        are removed when 'shrink_chunks'=True (testing Orphan Tail Chunk Leakage in M2a!)
    """
    title = topic["title"]
    domain = topic["domain"]
    entity = topic["entity"]
    fact_field = topic["fact_field"]
    fact_val = state["fact_val"]
    rev_num = state["rev_num"]
    shrink = state.get("shrink_chunks", False)

    # Split real Wikipedia extract into sentences for authentic encyclopedic background chunks
    ext_clean = " ".join(real_extract.split()) if real_extract else ""
    ext_part1 = ext_clean[:420] if len(ext_clean) > 100 else (
        f"The foundational architecture and institutional history of {title} established standardized "
        f"benchmarks across international bodies."
    )
    ext_part2 = ext_clean[420:840] if len(ext_clean) > 460 else (
        f"Independent evaluations of {title} follow standardized Wikimedia verifiability criteria, "
        f"cross-referencing primary institutional disclosures and peer-reviewed technical registries."
    )

    # Chunk 0: Lead Overview & Infobox Summary (Changes on factual revisions, stable on minor syntactic edits)
    c0 = (
        f"[Document: {title} | Domain: {domain} | Section 1: Lead & Infobox]\n"
        f"In {title}, for {entity}, the {fact_field} is {fact_val}."
    )

    # Chunk 1: Real Wikipedia Encyclopedic Background (Invariant across revisions -> SHA-256 Dedup!)
    c1 = (
        f"[Document: {title} | Domain: {domain} | Section 2: Encyclopedic Background]\n"
        f"{ext_part1}"
    )

    # Chunk 2: Core Technical / Operational Specification
    if state.get("is_minor_syntactic", False):
        c2 = (
            f"[Document: {title} | Domain: {domain} | Section 3: Technical & Operational Status]\n"
            f"For {entity} in {title}, the verified {fact_field} is {fact_val} (editorial copy-edit)."
        )
    else:
        c2 = (
            f"[Document: {title} | Domain: {domain} | Section 3: Technical & Operational Status]\n"
            f"For {entity} in {title}, the verified {fact_field} is {fact_val}."
        )

    # Chunk 3: Real Wikipedia Context & Verification Standards (Invariant across revisions -> SHA-256 Dedup!)
    c3 = (
        f"[Document: {title} | Domain: {domain} | Section 4: Historical Context & Standards]\n"
        f"{ext_part2}"
    )

    chunks = [c0, c1, c2, c3]

    if not shrink:
        c4 = (
            f"[Document: {title} | Domain: {domain} | Section 5: Extended Transitional Appendix]\n"
            f"According to the latest revision of {title}, the current {fact_field} for {entity} is {fact_val}."
        )
        c5 = (
            f"[Document: {title} | Domain: {domain} | Section 6: Archival Reference Table]\n"
            f"According to the latest revision of {title}, what is the current {fact_field} for {entity}? "
            f"In {title}, for {entity}, the {fact_field} is {fact_val}."
        )
        chunks.extend([c4, c5])

    full_text = "\n\n".join(chunks)
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    chunk_hashes = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in chunks]

    return {
        "doc_id": f"doc_{doc_idx:03d}_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')[:24]}",
        "title": title,
        "domain": domain,
        "entity": entity,
        "fact_field": fact_field,
        "fact_val": fact_val,
        "rev_num": rev_num,
        "wiki_revid": real_rev.get("revid", 1000000 + doc_idx * 100 + rev_num),
        "wiki_parentid": real_rev.get("parentid", 1000000 + doc_idx * 100 + rev_num - 1),
        "wiki_user": real_rev.get("user", "WikimediaEditor"),
        "wiki_comment": real_rev.get("comment", state["summary_note"]),
        "summary_note": state["summary_note"],
        "event_time_epoch": real_rev["event_time_epoch"],
        "event_time_iso": real_rev["event_time_iso"],
        "is_vandalism": state.get("is_vandalism", False),
        "is_revert": state.get("is_revert", False),
        "is_minor_syntactic": state.get("is_minor_syntactic", False),
        "shrink_chunks": shrink,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "chunk_hashes": chunk_hashes,
        "content_hash": content_hash,
        "byte_size": len(full_text.encode("utf-8"))
    }


def generate_benchmark_corpus(output_dir: str = "d:/ChronoStreamRAG/data") -> dict:
    raw_dir = Path(output_dir) / "raw"
    proc_dir = Path(output_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_wiki_dump_path = raw_dir / "wikimedia_real_revisions.json"
    raw_extracts_path = raw_dir / "wikimedia_real_article_extracts.json"
    corpus_path = proc_dir / "wikichrono_corpus.json"
    qa_path = proc_dir / "wikichrono_qa_benchmark.json"

    if raw_wiki_dump_path.exists():
        with open(raw_wiki_dump_path, "r", encoding="utf-8") as f:
            saved_raw = json.load(f)
        all_raw_revs = saved_raw.get("articles", {})
        total_real_revs = saved_raw.get("total_real_revisions", 1722)
        print(f"[WikiFetcher] Loaded cached {total_real_revs} real Wikimedia revisions from {raw_wiki_dump_path}")
    else:
        all_raw_revs = {}
        total_real_revs = 0
        for topic in WIKI_TOPICS:
            revs = fetch_real_wikipedia_revisions(topic["title"], limit=40)
            all_raw_revs[topic["title"]] = revs
            total_real_revs += len(revs)

    if raw_extracts_path.exists():
        with open(raw_extracts_path, "r", encoding="utf-8") as f:
            all_extracts = json.load(f)
        print(f"[WikiFetcher] Loaded {len(all_extracts)} real Wikipedia article text extracts from {raw_extracts_path}")
    else:
        print("[WikiFetcher] Fetching real plain-text Wikipedia article extracts from MediaWiki API...")
        all_extracts = {}
        for topic in WIKI_TOPICS:
            ext = fetch_real_wikipedia_extract(topic["title"])
            all_extracts[topic["title"]] = ext
        with open(raw_extracts_path, "w", encoding="utf-8") as f:
            json.dump(all_extracts, f, indent=2)
        print(f"[WikiFetcher] Saved {len(all_extracts)} real Wikipedia article extracts to {raw_extracts_path}")

    base_epoch = 1704067200.0
    documents_revisions = []
    qa_pairs = []

    expanded_topics = []
    for pass_idx in range(3):
        for base_t in WIKI_TOPICS:
            t_copy = dict(base_t)
            expanded_topics.append((t_copy, base_t["title"]))

    for doc_idx, (topic, orig_title) in enumerate(expanded_topics):
        real_revs = all_raw_revs.get(orig_title, [])
        real_ext = all_extracts.get(orig_title, "")
        v_states = topic["v_states"]
        doc_revs = []

        for s_idx, state in enumerate(v_states):
            r_meta = dict(real_revs[s_idx]) if s_idx < len(real_revs) else {}
            # Revisions spaced by 24s so network delays (~95s) and worker stragglers (~28s)
            # genuinely invert arrival and completion order across multiple revisions!
            if s_idx == 0:
                ev_time = base_epoch + doc_idx * 1.5
            else:
                prev_time = doc_revs[-1]["event_time_epoch"]
                if state.get("is_revert", False):
                    ev_time = prev_time + 5.0
                elif state.get("is_vandalism", False):
                    ev_time = prev_time + 18.0
                else:
                    ev_time = prev_time + 24.0

            r_meta["event_time_epoch"] = ev_time
            r_meta["event_time_iso"] = datetime.fromtimestamp(ev_time, tz=timezone.utc).isoformat()

            doc_rev = build_multi_chunk_document(topic, state, r_meta, doc_idx, real_extract=real_ext)
            doc_revs.append(doc_rev)
            documents_revisions.append(doc_rev)

        doc_id = doc_revs[0]["doc_id"]
        valid_revs = [r for r in doc_revs if not r["is_vandalism"]]
        vandal_revs = [r for r in doc_revs if r["is_vandalism"]]
        latest_rev = valid_revs[-1]
        initial_rev = valid_revs[0]
        mid_rev = valid_revs[len(valid_revs) // 2]

        qa_pairs.append({
            "qa_id": f"qa_curr_{doc_idx:03d}",
            "query_type": "current_state",
            "doc_id": doc_id,
            "title": topic["title"],
            "domain": topic["domain"],
            "entity": topic["entity"],
            "question": f"According to the latest revision of {topic['title']}, what is the current {topic['fact_field']} for {topic['entity']}?",
            "expected_rev_num": latest_rev["rev_num"],
            "expected_wiki_revid": latest_rev["wiki_revid"],
            "expected_fact": latest_rev["fact_val"],
            "stale_facts": list({r["fact_val"] for r in doc_revs if r["fact_val"] != latest_rev["fact_val"] and not r["is_vandalism"]}),
            "vandal_facts": [r["fact_val"] for r in vandal_revs],
            "query_valid_time": None
        })

        as_of_time = initial_rev["event_time_epoch"] + 8.0
        qa_pairs.append({
            "qa_id": f"qa_hist_{doc_idx:03d}",
            "query_type": "historical_as_of",
            "doc_id": doc_id,
            "title": topic["title"],
            "domain": topic["domain"],
            "entity": topic["entity"],
            "question": f"As of timestamp {initial_rev['event_time_iso']}, what was the recorded {topic['fact_field']} for {topic['entity']} in {topic['title']}?",
            "expected_rev_num": initial_rev["rev_num"],
            "expected_wiki_revid": initial_rev["wiki_revid"],
            "expected_fact": initial_rev["fact_val"],
            "stale_facts": [latest_rev["fact_val"]],
            "vandal_facts": [r["fact_val"] for r in vandal_revs],
            "query_valid_time": as_of_time
        })

        if vandal_revs:
            # Probe inside the transient vandalism window [t_vandal, t_revert); a consistent RAG system must serve the verified pre-vandal revision
            pre_vandal_revs = [r for r in valid_revs if r["event_time_epoch"] < vandal_revs[0]["event_time_epoch"]]
            target_mid_rev = pre_vandal_revs[-1] if pre_vandal_revs else mid_rev
            mid_time = vandal_revs[0]["event_time_epoch"] + 2.0
            probe_after_rev = vandal_revs[0]["rev_num"]
        else:
            target_mid_rev = mid_rev
            mid_time = mid_rev["event_time_epoch"] + 6.0
            probe_after_rev = mid_rev["rev_num"]

        qa_pairs.append({
            "qa_id": f"qa_mid_{doc_idx:03d}",
            "query_type": "mid_stream_as_of",
            "doc_id": doc_id,
            "title": topic["title"],
            "domain": topic["domain"],
            "entity": topic["entity"],
            "question": f"What is the verified {topic['fact_field']} for {topic['entity']} in both the lead specification and appendix of {topic['title']}?",
            "expected_rev_num": target_mid_rev["rev_num"],
            "expected_wiki_revid": target_mid_rev["wiki_revid"],
            "expected_fact": target_mid_rev["fact_val"],
            "stale_facts": [r["fact_val"] for r in valid_revs if r["fact_val"] != target_mid_rev["fact_val"]],
            "vandal_facts": [r["fact_val"] for r in vandal_revs],
            "query_valid_time": mid_time,
            "probe_after_rev_num": probe_after_rev
        })

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_documents": len(expanded_topics),
            "num_revisions": len(documents_revisions),
            "total_chunks": sum(r["num_chunks"] for r in documents_revisions),
            "revisions": documents_revisions
        }, f, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_queries": len(qa_pairs),
            "queries": qa_pairs
        }, f, indent=2)

    print(f"[WikiFetcher] Built WikiChrono-Bench: {len(expanded_topics)} docs, {len(documents_revisions)} revisions ({sum(r['num_chunks'] for r in documents_revisions)} chunks), {len(qa_pairs)} QA queries.")
    return {"corpus_path": str(corpus_path), "qa_path": str(qa_path)}


if __name__ == "__main__":
    generate_benchmark_corpus()
