"""test yolov8 model"""
from ultralytics import YOLO
from src.utils.config_utils import load_config

CONFIG = load_config()

class Yolo:
    """YoloV8 model class for object detection"""

    def __init__(self, model_path: str, print_logs: bool = False):
        """Initialize the YoloV8 model with a given path
        Args:
            model_path (str): Path to the YOLO model file model from scratch or pretrained
        Returns:
            None        
        """
        if print_logs:
            print(f"[INFO] Initializing YoloV8 model with path: {model_path}")
        self.model_path = model_path
        self.print_logs = print_logs
        self.model = YOLO(model_path)

    def get_model_info(self):
        """
        Get information about the YOLO model.
        Returns:
            dict: Model information including architecture, input size, and number of classes.
        """
        if self.print_logs:
            print("[INFO] Retrieving YoloV8 model information")
        model_info = self.model.info()
        return model_info

    def train(self, 
              data: str, 
              epochs: int = 3, 
              enable_checkpoint_saving: bool = False, 
              save_period: int = -1
              ):
        """
        Train the YOLO model on a specified dataset
        Args:
            data (str): Path to the dataset YAML configuration file
            epochs (int): Number of training epochs (default is 3)
        Returns:
            results: Training results
        """
        if self.print_logs:
            print(f"[INFO] Training YoloV8 model on dataset: {data} for {epochs} epochs")
        results = self.model.train(
                                    data=data,
                                    epochs=epochs,
                                    imgsz=640,
                                    patience=10, # Early stopping
                                    save=enable_checkpoint_saving,
                                    save_period=save_period,
                                    project="runs/train",
                                    )          
        return results


    def predict(self, 
                image_path: str, 
                show_res: bool = True, 
                save: bool = False
                ):
        """
        Perform object detection on an image.
        Args:
            image_path (str): Path to the input image.
            show (bool): If True, display the result image with bounding boxes (default is False).
            save (bool):  If True, save the result image to disk (default is False).
        
        """
        if self.print_logs:
            print(f"[INFO] Performing prediction on image: {image_path}")
        results = self.model(image_path, save=save, conf=0.3, iou=0.3, project="runs/detect", show_boxes=True, save_crop=True)
        if show_res:
            results[0].show()
        # if save:
        #     results[0].save(filename="output.jpg")
        return results

    def eval(self):
        """
        Evaluate the model's performance on the validation set specified in the dataset YAML file.
        Returns:
            results: Evaluation results containing metrics such as mAP, precision, recall, etc.
        """
        if self.print_logs:
            print("[INFO] Evaluating YoloV8 model on validation set")
        results = self.model.val()
        return results

    def export(self, format: str = "onnx"):
        """
        Export the model to a specified format
        Args:
            format (str): Format to export the model (default is 'onnx')
        Returns:
            str: File path to the exported model.
        """
        if self.print_logs:
            print(f"[INFO] Exporting YoloV8 model to format: {format}")
        success = self.model.export(format=format)
        return success

def main_fun():
    """Test the YoloV8 model class"""
    print("[INFO] Starting YoloV8 training and prediction test")

    # test_image_path = r"/teamspace/studios/this_studio/EgyptianNationalId/src/test/test_images/my_test_img3.jpeg"
    # test_image_path = r"/teamspace/studios/this_studio/EgyptianNationalId/DATA/valid/images/45_jpg.rf.772f20124092554b7d5b06e78b12d59f.jpg"
    test_image_path = r"E:\DATA SCIENCE\projects\dl\EgyptianNationalId\DATA\train\images\2_jpg.rf.1e15b1bf037758b9c1dc8650f8212260.jpg"


    # model_path = CONFIG["YOLO_PATH"]["YOLOV8n_SEG"]

    # model_path = CONFIG["YOLO_PATH"]["YOLOV8n_SEG_LAST"]
    # model_path = CONFIG["YOLO_PATH"]["YOLOV8n_SEG_BEST"]
    model_path =r"src\best.pt"


    print_logs = True

    yolo = Yolo(model_path=model_path, print_logs=print_logs)

    detection_results = yolo.predict(test_image_path, show_res=True, save=True)


    # yolo.train(
    #     # data=r"/teamspace/studios/this_studio/EgyptianNationalId/DATA/config.yaml",  
    #     data = CONFIG["PATH"]["train_config"],
    #     epochs=50,
    #     enable_checkpoint_saving=True,
    #     save_period=5
    # )

    # detection_results = yolo.predict(test_image_path, show_res=True, save=True)

    # return detection_results

def main():
    """Main function to run the test"""
    main_fun()

    # # print("Training Results:", train_results)
    # # print("Evaluation Results:", eval_results)
    # # print("Detection Results:", detection_results)
    # print("Export Success:", export_success)
if __name__ == "__main__":
    main()

# to run this: python -m src.train.train_yolo