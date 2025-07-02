from ultralytics import YOLO
import time
import streamlit as st
import cv2


def load_model(model_path):
    """
    Loads a YOLO object detection model.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model
