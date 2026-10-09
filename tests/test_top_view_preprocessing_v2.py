import csv
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from DeepWhiskerCuts.lib.top_view_preprocessing_v2 import (
    frame_transforms, map_points, prepare_poses, preprocess, transform_image,
)


class GeometryTests(unittest.TestCase):
    def test_known_rotation_and_reflection(self):
        matrices, angle = frame_transforms([320, 240], [320, 310])
        self.assertEqual(angle, 180.)
        np.testing.assert_allclose(map_points([[320, 240]], matrices['L']), [[305, 350]])
        np.testing.assert_allclose(map_points([[320, 240]], matrices['R']), [[304, 350]])
        np.testing.assert_allclose(map_points([[420, 220]], matrices['L']), [[205, 370]])
        np.testing.assert_allclose(map_points([[220, 220]], matrices['R']), [[204, 370]])

    def test_pixels_and_points_use_same_transform(self):
        worst_error = 0.
        for angle in [-179., -110., -35., 0., 47., 135., 179.]:
            nose = np.array([320., 240.]);snout = nose+70*np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle))])
            matrices, _ = frame_transforms(nose, snout)
            for side in ('L', 'R'):
                matrix=matrices[side]
                targets=np.array([[220.,300.],[230.,420.]])
                points=map_points(targets,np.linalg.inv(matrix))
                np.testing.assert_allclose(map_points(map_points(points,matrix),np.linalg.inv(matrix)),points,atol=1e-10)
                for point in points:
                    source=np.zeros((480,640,3),np.uint8)
                    pixel=np.rint(point).astype(int);cv2.circle(source,tuple(pixel),3,(255,255,255),-1)
                    output=transform_image(source,matrix,contrast=1.)[:,:,0]
                    moments=cv2.moments(output)
                    observed=np.array([moments['m10']/moments['m00'],moments['m01']/moments['m00']])
                    predicted=map_points([pixel],matrix)[0]
                    error=np.linalg.norm(observed-predicted);worst_error=max(worst_error,error)
                    self.assertLess(error,.25)
        print(f'Max raster-versus-coordinate landmark error: {worst_error:.6f} pixels')

    def test_pose_gaps(self):
        raw=np.tile([80.,40.,1.,80.,70.,1.],(10,1));raw[2,2]=0;raw[5:9,2]=0
        _,good,interpolated,valid=prepare_poses(raw,max_gap=1,smoothing_radius=1)
        self.assertTrue(interpolated[2]);self.assertFalse(valid[5:9].any())
        self.assertFalse(good[2]);self.assertTrue(valid[2])

    def test_video_advances_and_bad_pose_does_not_drop_frame(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp);video=root/'source.avi';dlc=root/'source.csv'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),40.,(160,120))
            self.assertTrue(writer.isOpened())
            for level in [30,60,90,120,150,180]: writer.write(np.full((120,160,3),level,np.uint8))
            writer.release()
            with dlc.open('w',newline='') as handle:
                w=csv.writer(handle);w.writerow(['scorer']+['model']*6)
                w.writerow(['bodyparts']+['Nose']*3+['Snout']*3)
                w.writerow(['coords','x','y','likelihood','x','y','likelihood'])
                for i in range(6): w.writerow([i,80,40,0 if i==2 else 1,80,70,1])
            config,metadata=preprocess(video,dlc,root/'outputs',max_gap=0,smoothing_radius=0,contrast=1.)
            self.assertEqual(config['frames_written'],6)
            with metadata.open() as handle: rows=list(csv.DictReader(handle))
            self.assertEqual([int(r['source_frame0']) for r in rows],list(range(6)))
            self.assertEqual(int(rows[2]['pose_valid']),0)
            for side in ('L','R'):
                output=root/'outputs'/('MasksourceL.avi' if side=='L' else 'MirrorsourceR.avi')
                cap=cv2.VideoCapture(str(output));means=[]
                while True:
                    ok,im=cap.read()
                    if not ok: break
                    means.append(im.mean())
                cap.release();self.assertEqual(len(means),6);self.assertEqual(means[2],0)
                self.assertTrue(means[0]<means[1]<means[3]<means[4]<means[5])


if __name__=='__main__': unittest.main(verbosity=2)
