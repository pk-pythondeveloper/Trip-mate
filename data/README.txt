TripMate Destination Data Pack
==============================

This folder contains 4 destination guide documents ingested by the
`search_destination_guide` RAG tool:

  - tokyo.txt
  - reykjavik.txt
  - bangkok.txt
  - barcelona.txt

Each document covers 5 sections: Visa & Entry, Best Time to Visit,
Local Customs, Packing Tips, and Safety & Health.

Chunking strategy used: one chunk per section per document (20 chunks).
See tripmate/rag/ingest.py and the README's "Design decisions" section.

Note: this is a simplified reference dataset created for a technical
assessment exercise. It is not authoritative, up-to-date travel or visa
advice.
