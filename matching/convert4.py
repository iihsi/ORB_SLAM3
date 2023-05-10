import open3d as o3d
import numpy as np
import cv2

def getHMatrix(p, n):
    n *= -1
    q = np.array([0, 0, p[0]*n[0]/n[2] + p[1]*n[1]/n[2] + p[2]])
    a = (q - p) / np.linalg.norm(q - p) #nに垂直な平面上の単位ベクトル
    c = np.cross(n, a)
    # R = np.matrix([a, c, -n]).T
    # t = (np.array([p]).T) - R.dot((np.array([p]).T))

    R = np.matrix([a, c, -n])
    t = (np.array([p]).T) - R.T.dot((np.array([p]).T))

    n = np.matrix([n])

    K = np.matrix([[385.881,  0, 322.597], 
                    [ 0, 385.881, 237.023], 
                     [ 0,  0,  1] ])
    # K = np.matrix([[643.135, 0, 644.328], [0, 643.135, 355.038], [0, 0, 1]])

    H = (K.dot(R - t.dot(n)/np.linalg.norm(p))).dot(K.I)

    return H


if __name__ == "__main__":
    path = "lab1"
    pcd = o3d.io.read_point_cloud("./matching/" + path + ".ply")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)) #法線マップ

    o3d.visualization.draw_geometries([pcd], point_show_normal = True)

    pcd_np = np.asarray(pcd.points) #点群データ
    normals = np.asarray(pcd.normals) #法線ベクトル
    np.savetxt("./matching/" + path + "_point.txt", pcd_np)
    np.savetxt("./matching/" + path + "_normals.txt", normals)

    num = 60156

    H = getHMatrix(pcd_np[num-1], normals[num-1])
    print(num)
    print("point cloud", pcd_np[num-1])
    print("normal", normals[num-1])
    print("H", H)

    #x = np.matrix([[643.135, 0, 644.328], [0, 643.135, 355.038], [0, 0, 1]]) * (np.array([pcd_np[num]]).T)
    x = np.matrix([[385.881, 0, 322.597], [0, 385.881, 237.023], [0, 0, 1]]).dot(np.array([pcd_np[num-1]]).T)
    u = x[0]/x[2]
    v = x[1]/x[2]
    res = H.dot(x)
    u1 = res[0]/res[2]
    v1 = res[1]/res[2]
    print("u,v", u, v)
    print("u1,v1", u1, v1)

    img = cv2.imread("./matching/" + path + "_Color.png", cv2.IMREAD_COLOR)
    height, width, channels = img.shape[:3]
    dst = cv2.warpPerspective(img, H, (width, height))
    cv2.imwrite("./matching/" + path + "_cvt4.png", dst)
