import matplotlib
from scipy.spatial import Delaunay
from skimage.draw import polygon
from skimage.color import rgba2rgb
import scipy.interpolate as skinterp
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os


def choose_pts(img, N=46):
    """
    This function was used to select the points on my images.
    """
    plt.imshow(img)
    x_coors = []
    y_coors = []


    for i in range(N):
        x, y = plt.ginput(1)[0]
        x_coors.append(x)
        y_coors.append(y)

        plt.close()
        plt.plot(x_coors, y_coors, '.r')
        plt.annotate(i, (x, y), xytext=(x + 50, y + 50))

        plt.imshow(img)

    return np.stack([x_coors, y_coors], axis=1)


def part_one():
    """ Choosing 46 points or 54 points
    where the "46" points matches the keypoints off the FEI dataset
    and the 54 points additionally specifies glasses"""

    # seema = plt.imread("lib/seemas.jpeg")
    # seema_points = choose_pts(seema, N=46)
    # np.save("out/seema_points46.npy", seema_points)

    # varsha = plt.imread("lib/varshas.jpeg")
    # varsha_points = choose_pts(varsha, N=46)
    # np.save("out/varsha_points46.npy", varsha_points)

    varun = plt.imread("lib/varuns.jpeg")
    varun_points = choose_pts(varun, N=46)
    np.save("out/varun_points46.npy", varun_points)

    varun_points = choose_pts(varun, N=54)
    np.save("out/varun_points54.npy", varun_points)

    amit = plt.imread("lib/amits.jpeg")
    amit_points = choose_pts(amit, N=54)
    np.save("out/amit_points54.npy", amit_points)

    eva = plt.imread("lib/eva.jpeg")
    eva_points = choose_pts(eva, N=54)
    np.save("out/eva_points54.npy", eva_points)

    varun_points = add_borders(np.load("out/varun_points54.npy"))
    amit_points = add_borders(np.load("out/amit_points54.npy"))

    avg_points = (varun_points + amit_points) / 2
    avg_triangles = Delaunay(avg_points)

    plt.imshow(varun)
    plt.triplot(varun_points[:,0], varun_points[:, 1, ], avg_triangles.simplices)
    # plt.show()


def tri_to_identity(tri_x, tri_y):
    """The matrix transforming a triangle's vertices to the identity triangle:
    ((0,0), (0,1), (1,0))"""

    a_x, b_x, c_x = tri_x
    a_y, b_y, c_y = tri_y
    return np.int0(np.array([
        [c_x - a_x, b_x - a_x, a_x],
        [c_y - a_y, b_y - a_y, a_y],
        [0,         0,         1]
    ]))

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def interp2(source_r, source_c, source_rgb, out_shape, mode):
    """Somewhat naive interpolation function, going channel-by-channel. """
    time_0 = time()
    width, height = out_shape
    # this gets confused if we don't have height == width
    output_grid = np.array([[c, h] for h in np.arange(height) for c in np.arange(width)])

    # grayscale image
    if len(source_rgb.shape) < 2:
        v = source_rgb[:]
        intp_v = skinterp.griddata((source_c, source_r), v, output_grid, method=mode).reshape(height, width)
        return normalize(intp_v)

    r = source_rgb[:, 0]
    g = source_rgb[:, 1]
    b = source_rgb[:, 2]

    intp_r = skinterp.griddata((source_c, source_r), r, output_grid, method=mode).reshape(height, width)
    intp_g = skinterp.griddata((source_c, source_r), g, output_grid, method=mode).reshape(height, width)
    intp_b = skinterp.griddata((source_c, source_r), b, output_grid, method=mode).reshape(height, width)
    print(f"total interp time is {time() - time_0} seconds")
    return np.dstack((intp_r, intp_g, intp_b))


def computeAffine(img, tri1_points, tri2_points, mode="nearest"):
    """Morphs img A to some other img B, using the vertices for a triangulation of A
    and the vertices for a triangulation of B.
     Interpolate the output to match the shape of img. """
    assert tri1_points.shape == tri2_points.shape
    num_tris = tri1_points.shape[0]

    # used for interpolation
    source_r = []
    source_c = []
    source_rgb = []

    for i in range(num_tris):
        tri1_vertices = tri1_points[i]
        tri2_vertices = tri2_points[i]

        tri1_verticies_x = tri1_vertices[:, 0]
        tri1_verticies_y = tri1_vertices[:, 1]
        tri2_verticies_x = tri2_vertices[:, 0]
        tri2_verticies_y = tri2_vertices[:, 1]

        identity_to_tri1 = tri_to_identity(tri1_verticies_x, tri1_verticies_y)
        identity_to_tri2 = tri_to_identity(tri2_verticies_x, tri2_verticies_y)

        # polygon works on a "(row, column)" system
        tri1_y, tri1_x = polygon(tri1_verticies_y, tri1_verticies_x)
        # tri2_y, tri2_x = polygon(tri2_verticies_y, tri2_verticies_x)

        tri1_matrix = np.vstack([tri1_x, tri1_y, np.ones_like(tri1_x)])
        tri1_to_identity = np.linalg.lstsq(identity_to_tri1, tri1_matrix, rcond=None)[0]
        tri1_to_tri2 = np.int0(np.matmul(identity_to_tri2, tri1_to_identity))

        tri1_to_tri2_x = tri1_to_tri2[0, :]
        tri1_to_tri2_y = tri1_to_tri2[1, :]
        source_r.extend(tri1_to_tri2_y)
        source_c.extend(tri1_to_tri2_x)

        source_rgb.extend(img[tri1_y, tri1_x].tolist())

    return interp2(source_r, source_c, np.array(source_rgb), (250, 300), mode)


def part_two():
    # Points were selected, and saved in part 1.
    seema = rgba2rgb(plt.imread("lib/seemas.jpeg"))
    varsha = rgba2rgb(plt.imread("lib/varshas.jpeg"))

    seema_points = add_borders(np.load("out/seema_points46.npy"))
    varsha_points = add_borders(np.load("out/varsha_points46.npy"))

    avg_points = np.int0(np.floor((seema_points + varsha_points) / 2))
    avg_triangles = Delaunay(avg_points)

    seema_tri_pts = seema_points[avg_triangles.simplices]
    varsha_tri_pts = varsha_points[avg_triangles.simplices]
    avg_tri_pts = avg_triangles.points[avg_triangles.simplices]

    seema2avg = computeAffine(seema, seema_tri_pts, avg_tri_pts)
    plt.imsave("out/seema2avg.jpeg", seema2avg)

    varsha2avg = computeAffine(varsha, varsha_tri_pts, avg_tri_pts)

    plt.imsave("out/varsha2avg.jpeg", varsha2avg)

    combined = (seema2avg + varsha2avg) / 2
    plt.imsave("out/seemavarsha.jpeg", combined)



def morph(im1, im2, im1_pts, im2_pts, tris, t):
    # assert 0 <= t <= 1
    weighted_avg_pts = t * im1_pts + (1 - t) * im2_pts

    im1_tri_pts = im1_pts[tris]
    im2_tri_pts = im2_pts[tris]
    avgt_tri_pts = weighted_avg_pts[tris]

    im1_to_avgt = computeAffine(im1, im1_tri_pts, avgt_tri_pts)
    im2_to_avgt = computeAffine(im2, im2_tri_pts, avgt_tri_pts)

    avgt = t * im1_to_avgt + (1 - t) * im2_to_avgt
    return avgt
    # plt.imshow(avgt)
    # plt.show()

def add_borders(img_pts, row=250, col=300):
    # Slams any "too-large" values down
    img_pts[:, 0] = np.minimum(img_pts[:, 0], row - 1)
    img_pts[:, 1] = np.minimum(img_pts[:, 1], col - 1)

    # attaches four corner points to image
    borders = np.array([
        [0, 0],
        [row - 1, 0],
        [0, col - 1],
        [row - 1, col - 1]
    ])
    return np.vstack([borders, img_pts])

def part_three():
    varun = rgba2rgb(plt.imread("lib/varuns.jpeg"))
    eva = rgba2rgb(plt.imread("lib/eva.jpeg"))
    seema = rgba2rgb(plt.imread("lib/seemas.jpeg"))
    varsha = rgba2rgb(plt.imread("lib/varshas.jpeg"))
    amit = rgba2rgb(plt.imread("lib/amits.jpeg"))

    varun_points54 = add_borders(np.load("out/varun_points54.npy"))
    eva_points54 = add_borders(np.load("out/eva_points54.npy"))
    seema_points46 = add_borders(np.load("out/seema_points46.npy"))
    varsha_points46 = add_borders(np.load("out/varsha_points46.npy"))
    amit_points54 = add_borders(np.load("out/amit_points54.npy"))

    varuneva_points = np.int0(np.floor((eva_points54 + varun_points54) / 2))
    varuneva_triangles = Delaunay(varuneva_points).simplices
    i = 0
    for t in np.linspace(0, 1, num=60):
        morphed_im = morph(varun, eva, varun_points54, eva_points54, varuneva_triangles, t)
        plt.imsave(f"out/eva2varun/frame_{i}.jpeg", morphed_im)
        i += 1

    seemavarsha_points = np.int0(np.floor((seema_points46 + varsha_points46) / 2))
    seemavarsha_triangles = Delaunay(seemavarsha_points).simplices
    i = 0
    for t in np.linspace(0, 1, num=60):
        morphed_im = morph(seema, varsha, seema_points46, varsha_points46, seemavarsha_triangles, t)
        plt.imsave(f"out/varsha2seema/frame_{i}.jpeg", morphed_im)
        i += 1

    amitvarun_points = np.int0(np.floor((varun_points54 + amit_points54) / 2))
    amitvarun_triangles = Delaunay(amitvarun_points).simplices
    i = 0
    for t in np.linspace(0, 1, num=60):
        morphed_im = morph(amit, varun, amit_points54, varun_points54, amitvarun_triangles, t)
        plt.imsave(f"out/varun2amit/frame_{i}.jpeg", morphed_im)
        i += 1


def part_four():
    # Number of images in the smiling_faces (FEI) dataset
    N = 100

    # calculate avg of keypoints
    sum_keypts = np.zeros((50, 2))
    for f in os.listdir("lib/smiling_faces"):
        # img = plt.imread(os.path.join("lib/smiling_faces", f))
        img_name = os.path.splitext(f)[0]

        img_pts_file = f"lib/smiling_faces_pts/{img_name}.pts"
        img_pts = np.loadtxt(img_pts_file, comments=("version:", "n_points:", "{", "}"))
        img_pts = add_borders(img_pts)
        sum_keypts += img_pts

    smile_keypts = np.int0(np.floor(sum_keypts / N))
    np.save("out/smile_points", smile_keypts)

    smile_tris = Delaunay(smile_keypts)
    smile_tri_pts = smile_keypts[smile_tris.simplices]

    sum_faces = np.zeros((300, 250))
    # Morph each image to have keypts_avg shape
    for f in os.listdir("lib/smiling_faces"):
        img = plt.imread(os.path.join("lib/smiling_faces", f))
        img_name = os.path.splitext(f)[0]


        img_pts_file = f"lib/smiling_faces_pts/{img_name}.pts"
        img_pts = np.loadtxt(img_pts_file, comments=("version:", "n_points:", "{", "}"))
        img_pts = add_borders(img_pts)
        img_tri_pts = img_pts[smile_tris.simplices]

        img2avg = computeAffine(img, img_tri_pts, smile_tri_pts)
        if img_name in ["1b", "10b", "100b"]:
            plt.imsave(f"out/{img_name}2avg.jpeg", img2avg, cmap='gray')

        sum_faces += img2avg

    smile_face = sum_faces / N
    plt.imsave("out/smile_face.jpeg", smile_face, cmap='gray')


    # warping my face to the average geometry
    varun = rgba2rgb(plt.imread("lib/varuns.jpeg"))
    varun_points = add_borders(np.load("out/varun_points46.npy"))
    varun_tri_pts = varun_points[smile_tris.simplices]
    varun2avg = computeAffine(varun, varun_tri_pts, smile_tri_pts, mode="linear")

    plt.imshow(varun2avg)
    plt.show()
    plt.imsave("out/varun2smile.jpeg", varun2avg)

    # Warping average face to my geometry
    smile2varun = computeAffine(smile_face, smile_tri_pts, varun_tri_pts)
    plt.imshow(smile2varun, cmap='gray')
    plt.show()
    plt.imsave("out/smile2varun.jpeg", smile2varun, cmap='gray')



def part_five():
    smile_face = plt.imread("out/smile_face.jpeg")
    smile_points = np.load("out/smile_points.npy")
    smile_tris = Delaunay(smile_points)
    smile_tri_pts = smile_points[smile_tris.simplices]

    varun = rgba2rgb(plt.imread("lib/varuns.jpeg"))
    varun_points = add_borders(np.load("out/varun_points46.npy"))
    varun_tri_pts = varun_points[smile_tris.simplices]


    exaggerated_tri_pts = varun_tri_pts + 0.3 * (varun_tri_pts - smile_tri_pts)
    varun2exaggerate = computeAffine(varun, varun_tri_pts, exaggerated_tri_pts)
    plt.imsave("out/varun_charicature.jpeg", varun2exaggerate)


def part_six():
    # Bells and Whistles
    # used to make the music video
    sequence = ["misato1", "asuka", "rei", "misato2", "misato3", "misato4", "kaworu", "evas", "shinji", "rei2"]
    for itm in sequence:
        pts = choose_pts(plt.imread(f"lib/{itm}.jpeg"), N=12)
        np.save(f"out/{itm}_pts", pts)

    for i in range(len(sequence) - 1):
        itm1 = sequence[i]
        itm2 = sequence[i + 1]
        try:
            os.mkdir(f"out/{itm1}_to_{itm2}")
        except:
            pass
        itm1_img = plt.imread(f"lib/{itm1}.jpeg")
        itm2_img = plt.imread(f"lib/{itm2}.jpeg")

        itm1_pts = add_borders(np.load(f"out/{itm1}_pts.npy"))
        itm2_pts = add_borders(np.load(f"out/{itm2}_pts.npy"))


        merge_pts = np.int0(np.floor((itm1_pts + itm2_pts) / 2))
        merge_triangles = Delaunay(merge_pts).simplices
        i = 0
        for t in np.linspace(0, 1, num=30):
            morphed_im = np.uint8(morph(itm1_img, itm2_img, itm1_pts, itm2_pts, merge_triangles, t))
            # print(morphed_im)
            plt.imsave(f"out/{itm1}_to_{itm2}/frame_{i}.jpeg", morphed_im)
            i += 1


def main():
    part_one()
    part_two()
    part_three()
    part_four()
    part_five()
    part_six()


if __name__ == '__main__':
    main()
