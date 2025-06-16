import pytest
import pymupdf
import justsdk

from span_marker import SpanMarkerModel
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from transformers import pipeline


class TestBasicEndToEnd:
    """
    End-to-end pipeline test for packages/modules
    """

    @pytest.fixture(scope="class")
    def models(self):
        return {
            "embedder": SentenceTransformer("all-MiniLM-L6-v2"),
            "ner": SpanMarkerModel.from_pretrained(
                "tomaarsen/span-marker-roberta-large-ontonotes5"
            ),
            "summarizer": pipeline(
                "summarization", model="facebook/bart-large-cnn", device=-1
            ),
        }

    def test_complete_pipeline(self, sample_pdf_agile_methodology, models):
        """Test PDF → NER → Topics → Summary pipeline"""

        # 1. Extract text from PDF
        doc = pymupdf.open(sample_pdf_agile_methodology)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()

        text_clean = text.strip().replace("\n", " ")
        justsdk.print_info(
            f"Extracted {len(text_clean)} characters", newline_before=True
        )

        # 2. Named Entity Recognition
        entities = models["ner"].predict(text_clean)
        orgs = [e["span"] for e in entities if e["label"] == "ORG"]
        locations = [e["span"] for e in entities if e["label"] == "GPE"]
        people = [e["span"] for e in entities if e["label"] == "PERSON"]

        justsdk.print_info(f"Found {len(entities)} entities:")
        print(f"  Organizations: {orgs[:3]}")
        print(f"  Locations: {locations[:3]}")
        print(f"  People: {people[:3]}")

        # 3. Topic Modeling
        sentences = [s.strip() for s in text_clean.split(".") if len(s.strip()) > 50]
        if len(sentences) >= 3:
            umap_model = UMAP(
                n_neighbors=2,
                n_components=2,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            hdbscan_model = HDBSCAN(min_cluster_size=2, metric="euclidean")

            topic_model = BERTopic(
                embedding_model=models["embedder"],
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=False,
            )

            topics, _ = topic_model.fit_transform(sentences[:8])
            unique_topics = len(set(topics))
            justsdk.print_info(
                f"Discovered {unique_topics} topics from {len(sentences)} sentences"
            )

        # 4. Text Summarization - Handle long documents by chunking
        def chunk_text(text, max_words=800):
            words = text.split()
            chunks = []
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                chunks.append(chunk)
            return chunks

        chunks = chunk_text(text_clean)
        justsdk.print_info(
            f"Split document into {len(chunks)} chunks for summarization"
        )

        # 4a. Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            max_len = min(100, len(chunk.split()) // 3)
            min_len = min(20, max_len // 2)

            summary = models["summarizer"](
                chunk, max_length=max_len, min_length=min_len, do_sample=False
            )
            chunk_summaries.append(summary[0]["summary_text"])
            justsdk.print_info(f"  Chunk {i + 1} summary: {summary[0]['summary_text']}")

        # 4b. Combine chunk summaries into final summary
        combined_summary = " ".join(chunk_summaries)

        # 4c. If we have multiple chunk summaries, summarize them again for final summary
        if len(chunk_summaries) > 1:
            final_max_len = min(150, len(combined_summary.split()) // 2)
            final_min_len = min(30, final_max_len // 3)

            final_summary = models["summarizer"](
                combined_summary,
                max_length=final_max_len,
                min_length=final_min_len,
                do_sample=False,
            )
            summary_text = final_summary[0]["summary_text"]
            print(f"Final summary: {summary_text}")
        else:
            summary_text = combined_summary

        print(f"Summary ({len(summary_text.split())} words): {summary_text}")

        # Assertions
        assert len(text_clean) > 100, "Should extract meaningful text"
        assert len(entities) > 0, "Should find entities"
        assert len(summary_text) > 0, "Should generate summary"

        justsdk.print_success("End-to-end pipeline completed successfully")
