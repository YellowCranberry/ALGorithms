import numpy as np
import time 



class Desi_Lin_Alg(object):
    
    def matrix_solve(self,A,b):
        s1 = time.time()
        X = np.linalg.solve(A,b)
        s2 =time.time()
        print(f"time_taken by linalg: {s2-s1}")
        return X
    
    
    def fdsubs(self,A,b):
        s1 = time.time()
        n = A.shape[0]
        X = np.zeros((n,1),dtype=float)
        for i in range(0,n):
            sum_val=0
            for j in range(0,i):
                sum_val+=(A[i,j]*X[j])
            X[i]=(b[i]-sum_val)/A[i,i]
        s2 = time.time()
        print(f"time_taken by fdsubs: {s2-s1}")
        return X

    def bdsubs(self,A,b):
        s1 = time.time()
        n = A.shape[0]
        X = np.zeros((n,1),dtype=float)
        for i in range(n-1,-1,-1):
            sum_val=0
            for j in range(i+1,n):
                sum_val+=(A[i,j]*X[j])
            X[i]=(b[i]-sum_val)/A[i,i]
        s2 = time.time()
        print(f"time_taken by fdsubs: {s2-s1}")
        return X

    def sge(self,A,b,luf):
        s1=time.time()
        n= len(A)
        A = A.astype(float)
        b = b.astype(float)
        l = np.zeros((n,n),dtype=float)
        for i in range(0,n):
            for j in range (i+1,n):
                l[j,i] = A[j,i]/A[i,i]
                A[j]-= (l[j,i]*A[i])
                if not luf:
                    b[j] -= (l[j,i]*b[i])
        np.fill_diagonal(l,1)
        s2 = time.time()
        print(f"time_taken by sge: {s2-s1}")
        # here luf is just a switch if you want just sge or using sge for lu decomposition
        if luf:
            return A,l
        return  A,b       

    def solve_using_lu(self,A,b):
        s1=time.time()
        U,L = self.sge(A,b,1)
        C = self.fdsubs(L,b)
        X = self.bdsubs(U,C)
        s2 =time.time()
        print(f"time_taken by solving lu decomposition: {s2-s1}")
        return X

    def solve_using_plu(self,A,b):
        s1 =time.time()
        P,L,U = self.plu_decomposition(A)
        b1 = P@b 
        print("p matrix\n",P,'\n')
        print("L matrix\n",L,'\n')
        print("U matrix\n",U,'\n')
        print("b matrix\n",b1,'\n')
        C = self.fdsubs(L,b1)
        X = self.bdsubs(U,C)
        s2 =time.time()
        print(f"time_taken by solving Plu decomposition: {s2-s1}")
        return X

    def plu_decomposition(self,A):
        s1=time.time()
        n = A.shape[0]
        A = A.astype(float)
        P = np.eye(n)
        l = np.zeros((n,n),dtype=float)
        U = A
        for i in  range(0,n):
            index=i
            for j in range(i+1,n):
                if abs(U[j,i])>abs(U[index,i]):
                    index=j                
            if abs(U[i,i]) < 1e-12:
                raise ValueError("Matrix is singular or nearly singular")
            U[[i, index]] = U[[index, i]]
            P[[i, index]] = P[[index, i]]
            if i > 0:
                l[[i, index], :i] = l[[index, i], :i]
            for j in range(i+1,n):
                l[j,i]=U[j,i]/U[i,i]
                U[j]-=(l[j,i]*U[i])
        np.fill_diagonal(l,1)
        return P,l,U    









if __name__=="__main__":

    dla = Desi_Lin_Alg()
    # A = np.random.randint(1,100,(4,4))
    # # A1 = np.tril(A)
    # b = np.random.randint(1,60,(4,1))
    # # print(dla.fdsubs(A1,b),end="\n")
    # # print(dla.matrix_solve(A1,b))
    # print(f"solving using LU \n{dla.solve_using_lu(A,b)}")
    # print(f"solving using linalg\n{dla.matrix_solve(A,b)}")
    # print(f"solve using plu \n{dla.solve_using_plu(A,b)}")
    # new_a, new_b =dla.sge(A,b,0)
    # print("\n\n so now our a is\n",A,f"and b is {b}\n\n")
    # print(f"now after doing structured gaussian elimination a is \n {new_a}\n\n b is{new_b} ")
    M = np.array([[1, 1, 1],
                  [1, 1, 3],
                  [2, 5, 8]])
    N =np.array([[2],
                 [4],
                 [10]])
    print(dla.solve_using_plu(M,N))
