import os
import json
import copy
import math

from PIL import Image

from collections import namedtuple

def rotate_point(x, y, a):
    ox, oy = 0.5, 0.5
    qx = ox + math.cos(a) * (x - ox) - math.sin(a) * (y - oy)
    qy = oy + math.sin(a) * (x - ox) + math.cos(a) * (y - oy)
    return qx, qy

# cervix type will be inferred from file name 1,2,3 or None if unable to infer
class AnnotatedCervixImage:
    def __init__(self, filepath, cervix_type, image_width, image_height, xmin, ymin, xmax, ymax ):
        self._filepath = filepath
        self._cervix_type = cervix_type
        self._image_width = image_width
        self._image_height = image_height
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    @property
    def filepath(self):
        return self._filepath
    @property
    def cervix_type(self):
        return self._cervix_type
    @property
    def image_width(self):
        return self._image_width
    @property
    def image_height(self):
        return self._image_height
    @property
    def xmin(self):
        return self._xmin
    @property
    def ymin(self):
        return self._ymin
    @property
    def xmax(self):
        return self._xmax
    @property
    def ymax(self):
        return self._ymax

    def with_filename(self, filename):
        n = copy.copy(self)
        n._filepath = filename
        return n

    def with_coords( self, xmin, ymin, xmax, ymax):
        n = copy.copy(self)
        n._xmin, n._ymin, n._xmax, n._ymax = xmin, ymin, xmax, ymax
        return n

    def with_size( self, sz ):
        width, height = sz
        n = copy.copy(self)
        n._image_width, n._image_height = width, height
        return n

    def with_rotate_coords(self, angle):
        n = copy.copy(self)
        theta = math.radians(angle)
        n._xmin, n._ymin = rotate_point(n._xmin, n._ymin, theta)
        n._xmax, n._ymax = rotate_point(n._xmax, n._ymax, theta)
        return n

    def ratio(self):
        return (self.ymax - self.ymin) / (self.xmax - self.xmin)

def read_annotations(path):
    """
    import sloth annotations json into list[AnnotatedCervixImage]
    """
    alist = []
    with open(path) as json_data:
        d = json.load(json_data)

        type_map = {
            'type_1': 1,
            'type_2': 2,
            'type_3': 3
        }

        for elem in d:
            filepath = os.path.abspath(elem['filename'])
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except OSError:
                print(" E Error opening file: %s. Ignoring" % filepath )
                continue

            cervix_type = None
            t = filepath.split('/')[-2].lower()
            if t in type_map:
                cervix_type = type_map[t]

            annotations = elem['annotations']
            if len(annotations) == 1:
                a = annotations[0]
                x0, y0 = a['x'], a['y']
                x1, y1 = x0 + a['width'], y0 + a['height']
                # normalize bounding box
                ra = AnnotatedCervixImage(filepath, cervix_type, width, height, x0/width, y0/height, x1/width, y1/height)
            else:
                ra = AnnotatedCervixImage(filepath, cervix_type, width, height, None, None, None, None)

            alist.append(ra)

    return alist


def save_annotations(alist, output_path):
    res = []
    for im in alist:
        w,h,x0,y0,x1,y1 = im.image_width, im.image_height, im.xmin, im.ymin, im.xmax, im.ymax
        x0,y0,x1,y1 = x0*w, y0*h, x1*w, y1*h
        elem = {
            "class" : "image",
            "filename" : im.filepath,
            "annotations" : [{
                "class" : "rect",
                "height" : y1 - y0,
                "width" : x1 - x0,
                "x" : x0,
                "y" : y0
            }]
        }
        res.append(elem)

    with open(output_path, 'w') as fp:
        json.dump(res, fp)
