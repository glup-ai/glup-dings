import numpy as np
import time

class Face:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.id = None
        self.last_seen = time.time()
        self.votes = []
        
    def set_id(self, id):
        self.id = id

    def update(self, face):
        self.last_seen = time.time()
        self.x = face.x
        self.y = face.y
        self.width = face.width
        self.height = face.height


class FaceTracker:

    def __init__(self):
        self.faces = []

    def feed(self, new_face):
        distances = []
        for face in self.faces:
            if(time.time() - face.last_seen > 2):
                distances.append(1000)
            else:
                dist = np.sqrt((new_face.x - face.x)**2 + (new_face.y - face.y)**2)
                distances.append(dist)
            
        if(len(distances)>0):
            closest_index = np.argmin(distances)
            clostest_dist = distances[closest_index]

            if clostest_dist<new_face.width*1.5:
                match = self.faces[closest_index]
                match.update(new_face)
                return self.faces[closest_index]
        
        new_face.set_id(len(self.faces))
        self.faces.append(new_face)
        return new_face
