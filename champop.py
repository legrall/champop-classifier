import os
import math
import random
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
import pickle

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.segmaps import SegmentationMapOnImage

from mrcnn.config import Config
from mrcnn import model as modellib, utils

BACKGROUNDS_PICKLE = 'data/backgrounds.pck'
BACKGROUNDS_FOLDER = 'data/dtd'
CARDS_PICKLE = 'data/cards.pck'
LABELS_PATH = 'data/labels.txt'
IMAGE_HEIGHT = IMAGE_WIDTH = 704


############################################################
#  Configurations
############################################################

class ChampopConfig(Config):
    '''Configuration for training on Playing Card Dataset'''
    name = 'champop'

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1


############################################################
#  Dataset
############################################################

class BackgroundGenerator():
    def __init__(self, backgrounds_pck=BACKGROUNDS_PICKLE, backgrounds_folder=None):
        if backgrounds_folder:
            self._images = []
            for f in glob(backgrounds_folder + '/images/banded/*.jpg'):
                    self._images.append(mpimg.imread(f))
        else:
            self._images = pickle.load(open(backgrounds_pck,'rb'))
        self._nb_images=len(self._images)
        # print("Nb of backgrounds loaded :", self._nb_images)
    def get_random(self, display=False):
        bg = self._images[random.randint(0,self._nb_images-1)]
        return bg


class CardGenerator():
    def __init__(self, pickle_path=CARDS_PICKLE):
        self._cards=pickle.load(open(pickle_path,'rb'))
        # self._cards is a dictionary where keys are card names (ex:'Kc') and values are lists of (img,hullHL,hullLR) 
        self._nb_cards_by_value={k:len(self._cards[k]) for k in self._cards}
        # print("Nb of cards loaded per name :", self._nb_cards_by_value)
        
    def get_random(self, card_name=None, display=False):
        if card_name is None:
            card_name= random.choice(list(self._cards.keys()))
        card, hull =self._cards[card_name][random.randint(0,self._nb_cards_by_value[card_name]-1)]
        return card, card_name, hull


class SceneGenerator():
    def __init__(self, card_pickle_path, backgrounds_folder, one_class=True, size=(),
                 labels=None, overlap_ratio=0.8):
        self.imgW = size[0]
        self.imgH = size[1]
        self.ratio = overlap_ratio
        self.labels = labels
        self.card_gen = CardGenerator(pickle_path=card_pickle_path)
        self.background_gen = BackgroundGenerator(backgrounds_folder=backgrounds_folder)
        self.resize_bg = iaa.Resize({"height": self.imgH, "width": self.imgW})
        self.seq = iaa.Sequential([
            iaa.Affine(scale=[0.65,1]),
            iaa.Affine(rotate=(-180,180)),
            iaa.Affine(translate_percent={"x":(-0.25,0.25),"y":(-0.25,0.25)}),
        ])


    def augment_card(self, card, hull):
        cardH, cardW, _ = card.shape
        decalX=int((self.imgW-cardW)/2)
        decalY=int((self.imgH-cardH)/2)

        img_card = np.zeros((self.imgH, self.imgW, 4), dtype=np.uint8)
        img_card[decalY:decalY+cardH,decalX:decalX+cardW,:] = card

        card = img_card[:,:,:3]
        hull = hull[:,0,:]

        card_kps_xy = np.float32([
            [decalX, decalY],
            [decalX + cardW, decalY],
            [decalX + cardW, decalY + cardH],
            [decalX, decalY + cardH]
        ])
        kpsoi_card = KeypointsOnImage.from_xy_array(card_kps_xy, shape=card.shape)

        # hull is a cv2.Contour, shape : Nx1x2
        kpsoi_hull = [ia.Keypoint(x=p[0]+decalX, y=p[1]+decalY) for p in hull.reshape(-1,2)]
        kpsoi_hull = KeypointsOnImage(kpsoi_hull,
                                      shape=(self.imgH, self.imgW, 3))

        # create polygon
        poly_hull = Polygon(kpsoi_hull.keypoints)

        # create empty segmentation map for classes: background and card
        segmap = np.zeros((card.shape[0], card.shape[1], 3), dtype=np.uint8)

        # draw the tree polygon into the second channel
        segmap = poly_hull.draw_on_image(
            segmap,
            color=(0, 255, 0),
            alpha=1.0, alpha_lines=0.0, alpha_points=0.0)

        # merge the two channels to a single one
        segmap = np.argmax(segmap, axis=2)
        segmap = segmap.astype(np.uint8)
        segmap = SegmentationMapOnImage(segmap, nb_classes=2, shape=card.shape)

        myseq = self.seq.to_deterministic()
        card_aug, segmap_aug = myseq(image=card, segmentation_maps=segmap)
        card_aug, kpsoi_aug = myseq(image=card, keypoints=kpsoi_card)

        return card_aug, kpsoi_aug, segmap_aug


    def kps_to_mask(self, kpsoi):
        poly_card = Polygon(kpsoi.keypoints)

        segmap = np.zeros((self.imgH, self.imgW, 3), dtype=np.uint8)

        # draw the tree polygon into the second channel
        segmap = poly_card.draw_on_image(
            segmap,
            color=(0, 255, 0),
            alpha=1.0, alpha_lines=0.0, alpha_points=0.0)

        # merge the two channels to a single one
        segmap = np.argmax(segmap, axis=2)
        card_mask = np.stack([segmap]*3,-1)

        return card_mask


    def get_random(self):
        bg = self.background_gen.get_random()
        bg = self.resize_bg.augment_image(bg)

        while True:
            card1, card_name1, hull1 = self.card_gen.get_random()
            card_aug1, kpsoi_aug1, segmap_aug1 = self.augment_card(card1, hull1)
            card_mask1 = self.kps_to_mask(kpsoi_aug1)

            card2, card_name2, hull2 = self.card_gen.get_random()
            card_aug2, kpsoi_aug2, segmap_aug2 = self.augment_card(card2, hull2)
            card_mask2 = self.kps_to_mask(kpsoi_aug2)

            # Handle superposition
            arr = segmap_aug1.get_arr_int()
            original_size = np.sum(arr)
            arr = np.where(card_mask2[:, :, 0], 0, arr)
            segmap_aug1 = SegmentationMapOnImage(arr, nb_classes=2, shape=bg.shape)
            new_size = np.sum(arr)

            final = np.where(card_mask1, card_aug1, bg)
            final = np.where(card_mask2, card_aug2, final)
            if new_size < original_size * self.ratio:
                continue
            break

        # create empty segmentation map for classes: background, tree, chipmunk
        mask_final = np.zeros((self.imgH, self.imgW, 2), dtype=np.uint8)
        mask_final[:, :, 0] = card_mask2[:, :, 0]
        mask_final[:, :, 1] = np.where(card_mask2[:, :, 0], 0, card_mask1[:, :, 0])
        # mask_final[:, :, 0] = segmap_aug1.get_arr_int()
        # mask_final[:, :, 1] = segmap_aug2.get_arr_int()

        # Add classes
        class_id = np.array([1, 1])
        # class_id = np.array([self.labels.index(card_name1) + 1,
        #                      self.labels.index(card_name2) + 1])

        return final, (mask_final, class_id)


class ChampopDataset(utils.Dataset):
    """Generates the image with playing cards synthetic dataset. The dataset consists of a background
    from textured images and two card images placed randomly on it.
    The images are generated on the fly. No file access required.
    """

    def load_scenes(self, count, card_pickle_path, backgrounds_folder,
                    labels, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        for i, card in enumerate(labels):
            self.add_class("cards", i+1, card)
        
        # Generate random scene images
        scene_gen = SceneGenerator(card_pickle_path=card_pickle_path, 
                                   backgrounds_folder=backgrounds_folder,
                                   labels=labels, size=(height, width))

        for i in range(count):
            image, masks = scene_gen.get_random()
            self.add_image("cards", image_id=i, path=None, image=image,
                           width=image.shape[1], height=image.shape[0], cards=masks)


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = np.array(info['image'])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cards":
            return info["cards"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask, class_ids = info['cards']
        
        return mask.astype(np.bool), class_ids.astype(np.int32)
