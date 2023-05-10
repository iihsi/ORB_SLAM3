import open3d as o3d
import numpy as np
import cv2

def getHMatrix(p, n):
    K = np.matrix([[385.881,  0, 322.597],
                    [ 0, 385.881, 237.023],
                    [ 0,  0,  1] ])

    sin = np.sin(180)
    cos = np.cos(180)

    # from virtual camera to real camera
    R = np.matrix([ [ 1.,   0.,    0.],
                    [ 0.,  cos,  -sin],
                    [ 0.,  sin,   cos] ])

    t = np.matrix([ [0.], 
                    [p[1]],
                    [p[2]] ])

    n = np.matrix([ [n[0]], 
                    [n[1]],
                    [n[2]] ])

    H = K * (R - t*n.T/p[2]) * K.I

    return H


if __name__ == "__main__":
    path = "lab1"
    pcd = o3d.io.read_point_cloud("./matching/" + path + ".ply")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)) #法線マップ

    #o3d.visualization.draw_geometries([pcd], point_show_normal = True)

    pcd_np = np.asarray(pcd.points) #点群データ
    normals = np.asarray(pcd.normals) #法線ベクトル
    np.savetxt("./matching/" + path + "_point.txt", pcd_np)
    np.savetxt("./matching/" + path + "_normals.txt", normals)

    num = 141939

    H = getHMatrix(pcd_np[num], normals[num])
    print(H)

    #x = np.matrix([[643.135, 0, 644.328], [0, 643.135, 355.038], [0, 0, 1]]) * (np.array([pcd_np[num]]).T)
    x = np.matrix([[385.881, 0, 322.597], [0, 385.881, 237.023], [0, 0, 1]]) * (np.array([pcd_np[num]]).T)
    u = x[0]/x[2]
    v = x[1]/x[2]
    res = H * x
    u1 = res[0]/res[2]
    v1 = res[1]/res[2]
    print(u, v)
    print(u1, v1)

    img = cv2.imread("./matching/" + path + "_Color.png", cv2.IMREAD_COLOR)
    height, width, channels = img.shape[:3]
    dst = cv2.warpPerspective(img, H, (640, 480))
    cv2.imwrite("./matching/" + path + "_cvt3.png", dst)
