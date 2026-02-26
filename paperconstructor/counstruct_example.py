from constructors import Arxiv

paper_id = "2502.10794 "
example_paper = Arxiv(paper_id="2502.10794")
example_paper.load()
example_paper.chunker(chunk_size=300)
example_paper.save_chunks(include_metadata=True)
example_paper.get_refs()
refs = example_paper.get_refs()
for ref in refs[:3]:
    ref_paper = Arxiv(paper_id=ref["id"])
    ref_paper.load()
    ref_paper.download_pdf(output_path=f"{ref['id']}.pdf")
    ref_paper.chunker(chunk_size=300)
    ref_paper.save_chunks(include_metadata=True)
