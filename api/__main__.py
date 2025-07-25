import shutil

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from http import HTTPStatus
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.input_processor import InputProcessor
from src.ner import Ner
from src.summary import Summary
from configs._constants import ROOT_PATH, DATA_DIR

app = FastAPI(
    title="Kairo Core",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".mp3", ".mp4"}


def save_upload_file(upload_file: UploadFile) -> Path:
    file_ext = Path(upload_file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_path = UPLOADS_DIR / upload_file.filename
    counter = 1
    while file_path.exists():
        stem = Path(upload_file.filename).stem
        file_path = UPLOADS_DIR / f"{stem}_{counter}{file_ext}"
        counter += 1

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path


def cleanup_file(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


@app.get("/")
async def root():
    return {"message": "Kairo Core is running", "status": "healthy"}


@app.post("/process/upload")
async def process_uploaded_file(file: UploadFile = File(...), cleanup: bool = True):
    """
    Upload and process a file (PDF, DOCX, MP3, MP4).

    Args:
        file: The uploaded file
        cleanup: Whether to delete the file after processing (default: True)

    Returns:
        Processed content from the file
    """
    file_path = None
    try:
        file_path = save_upload_file(file)
        relative_path = file_path.relative_to(ROOT_PATH)

        result = InputProcessor.process(relative_path)
        result["file_info"] = {
            "filename": file.filename,
            "size": file.size,
            "content_type": file.content_type,
        }

        return JSONResponse(
            content=result, status_code=result.get("status_code", HTTPStatus.OK)
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "error": str(e),
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    finally:
        if cleanup and file_path:
            cleanup_file(file_path)


@app.post("/process/batch")
async def process_batch_files(
    files: list[UploadFile] = File(...), cleanup: bool = True
):
    """
    Upload and process multiple files.

    Args:
        files: List of uploaded files
        cleanup: Whether to delete files after processing (default: True)

    Returns:
        List of processed content from each file
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Batch size exceeds limit"
        )

    results = []
    file_paths = []

    try:
        for file in files:
            file_path = save_upload_file(file)
            file_paths.append(file_path)
            relative_path = file_path.relative_to(ROOT_PATH)

            result = InputProcessor.process(relative_path)
            result["file_info"] = {
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type,
            }
            results.append(result)

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "error": str(e),
                "partial_results": results,
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    finally:
        if cleanup:
            for file_path in file_paths:
                cleanup_file(file_path)


@app.post("/ner/extract")
async def extract_entities(data: dict):
    """
    Extract named entities from text.

    Expected input: {"texts": "text string" or ["text1", "text2", ...]}
    """
    try:
        texts = data.get("texts")
        if not texts:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Missing 'texts' field in request body",
            )

        result = Ner.extract_entities(texts)
        return JSONResponse(
            content=result, status_code=result.get("status_code", HTTPStatus.OK)
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": f"Failed to extract entities: {str(e)}",
                "data": None,
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@app.post("/summary/abstractive")
async def generate_abstractive_summary(data: dict):
    """
    Generate abstractive summary from text.

    Expected input: {"texts": "text string" or ["text1", "text2", ...]}
    """
    try:
        texts = data.get("texts")
        if not texts:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Missing 'texts' field in request body",
            )

        result = Summary.abstract_summarize(texts)
        return JSONResponse(
            content=result, status_code=result.get("status_code", HTTPStatus.OK)
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": f"Failed to generate abstractive summary: {str(e)}",
                "data": None,
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@app.post("/summary/extractive")
async def generate_extractive_summary(data: dict):
    """
    Generate extractive summary from text.

    Expected input: {"texts": "text string" or ["text1", "text2", ...]}
    """
    try:
        texts = data.get("texts")
        if not texts:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Missing 'texts' field in request body",
            )

        result = Summary.extract_summarize(texts)
        return JSONResponse(
            content=result, status_code=result.get("status_code", HTTPStatus.OK)
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": f"Failed to generate extractive summary: {str(e)}",
                "data": None,
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@app.post("/pipeline/full")
async def full_pipeline(file: UploadFile = File(...), cleanup: bool = True):
    """
    Complete pipeline: Upload file -> Process -> Extract entities -> Generate summaries.

    Args:
        file: The uploaded file
        cleanup: Whether to delete the file after processing (default: True)

    Returns:
        Complete analysis including content, entities, and summaries
    """
    file_path = None
    try:
        file_path = save_upload_file(file)
        relative_path = file_path.relative_to(ROOT_PATH)

        process_result = InputProcessor.process(relative_path)

        if process_result["status"] != "success":
            return JSONResponse(
                content=process_result, status_code=process_result.get("status_code")
            )

        content = process_result["content"]

        ner_result = Ner.extract_entities(content)

        abs_summary_result = Summary.abstract_summarize(content)
        ext_summary_result = Summary.extract_summarize(content)

        pipeline_result = {
            "status": "success",
            "status_code": HTTPStatus.OK,
            "message": "Full pipeline completed successfully",
            "data": {
                "file_info": {
                    "filename": file.filename,
                    "size": file.size,
                    "content_type": file.content_type,
                },
                "content": content,
                "entities": ner_result.get("data", {}),
                "abstractive_summary": abs_summary_result.get("data", {}),
                "extractive_summary": ext_summary_result.get("data", {}),
            },
        }

        return JSONResponse(content=pipeline_result)

    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": f"Pipeline failed: {str(e)}",
                "data": None,
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    finally:
        if cleanup and file_path:
            cleanup_file(file_path)


@app.get("/uploads/list")
async def list_uploads():
    try:
        files = [f.name for f in UPLOADS_DIR.iterdir() if f.is_file()]
        return {"files": files, "count": len(files)}
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@app.delete("/uploads/cleanup")
async def cleanup_uploads():
    try:
        deleted_count = 0
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1

        return {"message": f"Deleted {deleted_count} files"}
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
