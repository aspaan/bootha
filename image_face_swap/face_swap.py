from typing import Any, List, Callable, Optional
import insightface
import threading
import cv2
import os
from typing import Any
from gfpgan.utils import GFPGANer
from insightface.app.common import Face
import numpy
import logging
import json
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s [%(levelname)s]: %(message)s",  # Set the log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Set the date format
)

Face = Face
Frame = numpy.ndarray[Any, Any]

THREAD_SEMAPHORE = threading.Semaphore()
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
FACE_ANALYSER = None
FACE_ENHANCER = None

def face_analyzer() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l')
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER
def face_swap_model() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../model/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=get_swap_providers)
    return FACE_SWAPPER

def enhance_model() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../model/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_gfpgan_device())
    return FACE_ENHANCER

def get_swap_providers() -> list[str]:
    execution_provider = os.environ.get("EXECUTION_PROVIDER")
    if execution_provider == 'gpu':
        return ['CUDAExecutionProvider','CPUExecutionProvider']
    if execution_provider == 'coreml':
        return ['CoreMLExecutionProvider','CPUExecutionProvider']
    return ['CPUExecutionProvider']

def get_gfpgan_device() -> str:
    execution_provider = os.environ.get("EXECUTION_PROVIDER")
    if execution_provider == 'gpu':
        return 'cuda'
    if execution_provider == 'coreml':
        return 'mps'
    return 'cpu'

def process_image(source_path: str, target_path: str, output_path: str, enhance: bool) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    many_faces = get_many_faces(target_frame)
    if many_faces:
        for target_face in many_faces:
            target_frame = swap_face(source_face, target_face, target_frame)
            if enhance:
                target_frame = enhance_face(target_face, target_frame)
    cv2.imwrite(output_path, target_frame)

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    result = face_swap_model().get(temp_frame, target_face, source_face, paste_back=True)
    return result

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]

    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, temp_face = enhance_model().enhance(
                temp_face,
                paste_back=True
            )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face

    return temp_frame



def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        return face_analyzer().get(frame)
    except ValueError:
        return None