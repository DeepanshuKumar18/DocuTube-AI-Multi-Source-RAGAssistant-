from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


# YouTube Loader
def load_youtube_docs(url: str):

    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=30,
            language=["en", "en-IN", "hi"],
        )

        docs = loader.load()

        # Empty transcript check
        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError("Transcript is empty or unreadable")

        return docs

    except (NoTranscriptFound, TranscriptsDisabled):
        raise ValueError(
            "❌ No transcript available for this video. " "Please try another video."
        )

    except Exception:
        raise ValueError(
            "❌ Failed to load YouTube transcript. "
            "The video may not support transcripts."
        )


# PDF Loader
def load_pdf_docs(path: str):
    """
    Load PDF safely.
    Handles:
    - Empty PDF
    - Image-only PDF
    """

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        # Empty / scanned PDF check
        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError(
                "❌ This PDF contains no readable text. "
                "Please upload a text-based PDF."
            )

        return docs

    except Exception:
        raise ValueError(
            "❌ Failed to read the PDF file. "
            "The file may be corrupted or unsupported."
        )
