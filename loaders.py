from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


# YouTube Loader
def load_youtube_docs(url: str):
    
    # Loads transcript from a YouTube video.
    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=30,
            language=["en", "en-IN", "hi"],
        )

        docs = loader.load()

        # Transcript exists but contains no readable text
        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError("Transcript content is empty or unreadable.")

        return docs

    # captions are disabled or missing
    except (NoTranscriptFound, TranscriptsDisabled):
        raise ValueError(
            "Transcript is not available for this video."
        )

    # Invalid URL, network failure, or unsupported video
    except Exception :
        raise ValueError(
            "Unable to load YouTube transcript. "
            "Please check the video URL and try again."
        )


# PDF Loader
def load_pdf_docs(path: str):
    # Loads text-based PDFs only.
    # Scanned or image-only PDFs are not supported.

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        # PDF parsed successfully but has no readable text
        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError(
                "This PDF does not contain readable text. "
                "Please upload a text-based PDF."
            )

        return docs

    # corrupted file, or unsupported format
    except Exception:
        raise ValueError(
            "Unable to read the PDF file. "
            "The file may be corrupted or unsupported."
        ) 
