import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


x = np.arange(0,129,21.5)
y = np.arange(0,193.5,21.5)


# xv, yv = np.meshgrid(x, y)
# plt.plot(xv, yv, marker='o', color='k', linestyle='none')
# plt.show()

objpts = []
objpts2 = []
imgpts = []

for i in x:
    for j in y:
        objpts.append([j,i])
objpts = np.array(objpts)

for i in x:
    for j in y:
        objpts2.append([j,i,1])
objpts2 = np.array(objpts2)


# print(objpts)

# H_stack = []
def get_H(img_name,objpts):
    name = str(img_name) + '.jpg'
    img  = cv.imread('./calibration_images/'+name)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 

    ret,corners = cv.findChessboardCorners(gray,(9,6),None)
    # imgpts = []
    # print('corners=\n',corners)
    # # if chessboard corners are detected
    if ret == True:
        # print('true')
        img = cv.drawChessboardCorners(img, (9,6), corners,ret)
        cv.imwrite('./img_corners/corner '+ name, img)

    cv.destroyAllWindows()  

    H,status = cv.findHomography(objpts,corners)
    # print('H=\n',H)

    H = H/H[2,2]
    # h1 = H.T[0]
    # h2 = H.T[1]
    # h3 = H.T[2]

    # print('h1 =\n',h1)
    # print('h2 =\n',h2) 
    # print('h3 =\n',h3)
    return H,corners

def get_Vtemp(H):

    h1 = H.T[0]
    h2 = H.T[1]
    h3 = H.T[2]
    v12 = np.array([h1[0]*h2[0], h1[0]*h2[1]+h1[1]*h2[0], h1[1]*h2[1], h1[2]*h2[0]+h1[0]*h2[2], h1[2]*h2[1]+h1[1]*h2[2], h1[2]*h2[2]]).reshape(6,1)
    
    # print('\nv12 =\n', v12)

    v11 = np.array([h1[0]*h1[0], h1[0]*h1[1]+h1[1]*h1[0], h1[1]*h1[1], h1[2]*h1[0]+h1[0]*h1[2], h1[2]*h1[1]+h1[1]*h1[2], h1[2]*h1[2]]).reshape(6,1)
    # print('\nv11 =\n', v11)


    v22 = np.array([h2[0]*h2[0], h2[0]*h2[1]+h2[1]*h2[0], h2[1]*h2[1], h2[2]*h2[0]+h2[0]*h2[2], h2[2]*h2[1]+h2[1]*h2[2], h2[2]*h2[2]]).reshape(6,1)
    # print('\nv22 =\n', v22)

    v12_T = v12.T
    # print(np.shape(v12_T))

    v1_2 = (v11-v22).T
    # print('\nv11_T =\n', v11_T)

    # print('\n(v11-v22).T =\n',v1_2)
    V_temp = np.concatenate((v12_T,v1_2),axis=0)
    # print('\nV_temp\n=',V_temp)
    return V_temp


def get_V(objpts):
    V = []
    for i in range(1,14):
        H,corners = get_H(i,objpts)
        V_temp = get_Vtemp(H)
        V.append(V_temp)
    V = np.array(V).reshape(2*13,6)
    # print(np.shape(V))
    return V

def get_A(V):
    U,S,V_final=np.linalg.svd(V)
    b = V_final[np.argmin(S)]
    # b = b.reshape(6,1)
    # print('\nb=\n',b)

    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,0] = b[3]
    B[2,1] = b[4]
    B[2,2] = b[5]
    # print('B=\n',B)

    v0 = (b[1]*b[3]-b[0]*b[4])/(b[0]*b[2]-b[1]**2)

    # print('v0 =',v0)

    lamb = (b[5] - (b[3]**2 + v0*(b[1]*b[3]-b[0]*b[4]))/b[0])
    # print('\nlambda = ',lamb)
    # print(lamb/b[0])
    alpha = np.sqrt(lamb/b[0])
    # print('alpha=',alpha)
    beta = np.sqrt(lamb*b[0]/(b[0]*b[2]-b[1]*b[1]))
    # print('beta=',beta)
    gamma = (-b[1]*(alpha**2)*beta)/lamb
    # print('gamma=',gamma)
    u0 = (gamma*v0)/beta - b[3]*(alpha**2)/lamb
    # print('u0=',u0)

    A = np.zeros((3,3),np.float32)

    A[0][0] = alpha
    A[0][1] = gamma
    A[0][2] = u0
    A[1][1] = beta
    A[1][2] = v0
    A[2][2] = 1

    # print('\nA=\n',A)
    return A,lamb,u0,v0,alpha,beta

# get_V(1)
def get_P(A,H):
    # print('H=\n',H)
    lamb = 1/np.linalg.norm(np.matmul(np.linalg.inv(A),H[0]),2)
    r1 = lamb * np.matmul(np.linalg.inv(A),H[:,0])
    r2 = lamb * np.matmul(np.linalg.inv(A),H[:,1])
    r3 = np.cross(r1,r2)
    t  = lamb * np.matmul(np.linalg.inv(A),H[:,2])

    # print('\nr1=\n',r1) 
    # print('\nr2=\n',r2) 
    # print('\nr3=\n',r3) 
    # print('\nt=\n',t) 
             
    P = np.concatenate((r1,r2,t),axis = 0).reshape(3,3)
    # print('\nP=\n',P)

    P = P.T
    # print('\nP=\n',P)
    return P

def get_error(A,P,objpts,imgpts):
    a_p = np.matmul(A,P)
    mhat = np.matmul(a_p,objpts.T)
    error = imgpts.reshape((54,2)) - mhat.T[:,0:2]
    error = np.linalg.norm(error,axis=0,ord=2)
    return error  


def get_k(corner_stack,A,P):

    d = []
    D = []
    for j in range(len(objpts)):
        i = objpts[j]
        model2d = np.array([i[0], i[1], 1])
        model2d = np.reshape(model2d, (3, 1))
        # print(model2d)
        model3d = np.array([i[0], i[1],0, 1])
        model3d = np.reshape(model3d, (4, 1))
        # print(model3d)
        xyz = np.matmul(P,model2d)
        x = xyz[0]/xyz[2]
        y = xyz[1]/xyz[2]
        sq = x**2 + y**2

        ART = np.matmul(A,P)
        # print(ART)
        uvw = np.matmul(ART,model2d)
        u = uvw[0]/uvw[2]
        v = uvw[1]/uvw[2]
        # print(ART)  
        # print(corner_stack[j][0])
        u_ = corner_stack[j][0][0]
        v_ = corner_stack[j][0][1]
        u0 = A[0,2]
        v0 = A[1,2]
        D.append([sq*(u-u0), sq*sq*(u-u0)])
        D.append([sq*(v-v0) ,sq*sq*(v-v0) ])

        d.append([u_ - u])
        d.append([v_ - v])
        # print(D)
    D = np.reshape(np.array(D),(108,2))
    d = np.reshape(np.array(d),(108,1))
    # print(D.shape)
    # print(d.shape)
    DD = np.matmul(D.T,D)
    DDinv = np.linalg.inv(DD)
    DDD = np.matmul(DDinv,D.T)
    K = np.matmul(DDD,d)
    # print("k",K)

    return K

# for i in f:
#     k = distortion(i,A)


H_stack = []
corner_stack = []

V = get_V(objpts)
# proj_error = []

for i in range(1,14):
    name = str(i)+'.jpg'
    img1 = cv.imread('./calibration_images/'+ name)
    H,corners = get_H(i,objpts)
    corner_stack.append(corners)
    # print('H' +str(i)+ '=\n',H)
    # V = V_temp 
    # print(np.shape(V_temp))
    H_stack.append(H)
    A,lamb,u0,v0,alpha,beta = get_A(V)
    P = get_P(A,H) 

    e = get_error(A,P,objpts2,corners)
    e = np.sum(e)
print('\nA=\n',A) 
# print('\nP=\n',P) 

e = e/(9*6*13)
print('\nProjection Error=\n',e) 
# print('\nDone!')
   
for i in range(1,14):  
    # print(i)
    # print('se of p=',P.shape)
    k = get_k(corner_stack[i-1],A,P)
print('\nk=\n',k)
k0 = float(k[0])
k1 = float(k[1])
D = [k0,k1,0,0]
D = np.array(D,np.float32)
# print('\nD\n=',D)

for i in range(1,14):
    img = cv.imread('./calibration_images/'+ str(i) + '.jpg')
    undistorted_img = cv.undistort(img,A,D)
    cv.imwrite('./fixed_images/Undistorted '+str(i) + '.jpg', undistorted_img)


    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('Original Image ' + str(i))
    # plt.axis('off')
    # plt.imshow(img)

    # plt.subplot(1,2,2)
    # plt.title('Undistorted Image ' + str(i))
    # plt.axis('off')
    # plt.imshow(undistorted_img)

    # plt.show()

# cv.imshow('img1',img1)
# cv.imshow('undistorted_img1',undistorted_img)
# cv.waitKey(0)

