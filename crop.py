import cv2
class MouseCrop:
    def __init__(self, image):
        print("in init")   
        self.image = image
        self.coordinates = []
        self.width = 0
        self.height = 0
        self.cropping = False
    def handle_clicks(self, event, x, y, flags, params):
        # below print statement occurs multiple times
        # print('handle clicks')    
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinates = [(x, y)]
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            self.width = abs(x - self.coordinates[0][0])
            self.height = abs(y - self.coordinates[0][1])
        elif event == cv2.EVENT_LBUTTONUP and self.cropping:
            self.coordinates.append((x, y))
            self.cropping = False
            self.width = abs(self.coordinates[1][0] - self.coordinates[0][0])
            self.height = abs(self.coordinates[1][1] - self.coordinates[0][1])
            cv2.rectangle(self.image, self.coordinates[0], self.coordinates[1], (0, 255, 0), 2)
            cv2.putText(self.image, f"Width: {self.width}px", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(self.image, f"Height: {self.height}px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Image', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.crop_image()
    def crop_image(self):
        print("in crop image")
        if self.width > 0 and self.height > 0:
            cropped_image = self.image[self.coordinates[0][1]: self.coordinates[0][1] + self.height,
                                       self.coordinates[0][0]: self.coordinates[0][0] + self.width]
            cv2.imshow("Cropped Image", cropped_image)
            cv2.imwrite('mongo/current.jpg', cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Please select the image region to crop")
    def show_image(self):
        print('in show image')
        cv2.imshow("Image", self.image)
        cv2.setMouseCallback("Image", self.handle_clicks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == "__main__":
    # path = "E:\\karthik_drugs_recomendations\\mongo\\images\\prescriptions\\1.jpg"
    path = "mongo/20.jpg"
    image = cv2.imread(path)
    if image is not None:
        mouse_click_handler = MouseCrop(image)
        mouse_click_handler.show_image()
    else:
        print("Could not load image")