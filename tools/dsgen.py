import os,sys
import fnmatch
from stat import *
import random
import shutil
import argparse

def walktree(top, callback):
    for f in os.listdir(top):
        pathname = os.path.join(top,f)
        mode = os.stat(pathname).st_mode
        if S_ISDIR(mode):
            walktree(pathname, callback)
        elif S_ISREG(mode):
            callback(pathname)
        else:
            return

class xdataset(object):
    def __init__(self, testpercent=10, storedir='.'):
        self._trainset = []
        self._valset = []
        self._inferset = []
        self._testpercent = testpercent
        self._storedir = storedir
        random.seed()

    def _process(self, pathname):
        if not fnmatch.fnmatch(pathname,"*.jpg") and not fnmatch.fnmatch(pathname,"*.jpeg"):
            return 

        ann = os.path.splitext(pathname)[0] + '.xml'
        if not os.path.exists(ann):
            self._inferset.append(pathname)
        else:
            v = random.randint(0, 100)
            if v >= self._testpercent:
                self._trainset.append(pathname)
            else:
                self._valset.append(pathname)

    def __call__(self, pathname):
        self._process(pathname)

    def buildJPEGImages(self):
        dir = os.path.join(self._storedir, "JPEGImages")
        if not os.path.exists(dir):
            os.mkdir(dir)

        for f in self._trainset + self._valset + self._inferset:
            shutil.copy(f, dir)

    def buildAnnotations(self):
        dir = os.path.join(self._storedir, "Annotations")
        if not os.path.exists(dir):
            os.mkdir(dir)

        for f in self._trainset + self._valset:
            ann = os.path.splitext(f)[0] + '.xml'
            shutil.copy(ann, dir)

    def buildImageSets(self):
        dir = os.path.join(self._storedir, "ImageSets")
        if not os.path.exists(dir):
            os.mkdir(dir)

        dir = os.path.join(dir, "Main")
        if not os.path.exists(dir):
            os.mkdir(dir)

        for filename, dset in [("trainval.txt", self._trainset),
                               ("test.txt", self._valset),
                               ("inference.txt", self._inferset)]:
            file = open(os.path.join(dir, filename), "w")
            for x in dset:
                file.write(os.path.splitext(os.path.basename(x))[0] + "\n")

            file.close()
    
    def build(self):
        top = self._storedir
        if not os.path.exists(top):
            os.mkdir(top)

        self.buildJPEGImages()
        self.buildAnnotations()
        self.buildImageSets()

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def inferset(self):
        return self._inferset

parser = argparse.ArgumentParser(description="Tool to build dataset")
parser.add_argument('--src', dest='srcdir', action='store',
                    required=True,
                    help='The directory of source data')
parser.add_argument('--dest', dest='destdir', action='store',
                    required=True,
                    help='The directory for dataset storage')
parser.add_argument('--testpercent', dest='testpercent', action='store',
                    default=10,
                    help='The ratio of dataset for test')
args = parser.parse_args()
xf = xdataset(testpercent=args.testpercent,
              storedir=args.destdir)
walktree(args.srcdir, xf)
xf.build()
