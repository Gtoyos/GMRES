#
#		--- Obligatorio 1 - Implementacion del metodo GMRES ---
#	Autores: Toyos, Mangado, Piñeyro y Fasanello
#		
#
#
import time				#Pruebas de tiempo
import os				#Fucniones del OS. Para guardar imagenes
import numpy as np 			#Funciones y estructuras para trabajar con matrices
import math as m			#Funciones matematicas basicas
import random				#Funciones generadoras de numeros aleatorios
import matplotlib.pyplot as plt		#Funciones para graficar 
from scipy.linalg import hilbert	#Matriz Hilbert
from scipy import sparse		#Matrices esparsas
random.seed()

#Retorna una matriz magica de dimension n
def magic(N):
	magic_square = np.zeros((N,N), dtype=int)
	n = 1
	i, j = 0, N//2
	while n <= N**2:
    		magic_square[i, j] = n
    		n += 1
    		newi, newj = (i-1) % N, (j+1)% N
    		if magic_square[newi, newj]:
        		i += 1
    		else:
        		i, j = newi, newj
	return(magic_square)

#Dado un vector (a,b) devuelve un vector (c,s) con los parametros a aplicar una
#Rotacion de Givens sobre (a,b) que anule su segunda entrada: G^T*(a,b)=(r,0)
def givensParam(X):
	if (X[1]==0):
		return np.array([1,0])
	if (X[0]==0):
		return np.array([0,1])
	if (abs(X[0])>abs(X[1])):
		t = -X[0]/X[1]
		s = 1/m.sqrt(1+t*t)
		return np.array([s*t,s])
	else:
		t = -X[1]/X[0]
		c = 1/m.sqrt(1+t*t)
		return np.array([c,c*t])
#Algoritmo a daptado de G. Golub, Matrix Combinatronics (pg 240)

#Dada una matriz de hessenberg H (m+1)*m, retorna (Q,R) tal que R=Q^t*H. Es decir, su descomposicion QR
def hessenbergQR(H):
	#print(H.shape[0],H.shape[1])
	Q = np.identity(H.shape[1]+1)	
	for i in range(H.shape[1]):
		X = np.array([H[i,i],H[i+1,i]])
		P = givensParam(X)
		G = np.identity(H.shape[1]+1)
		G[i,i] = P[0]
		G[i+1,i+1] = P[0]
		G[i,i+1] = -P[1]
		G[i+1,i] = P[1]
		Q = np.matmul(G,Q)	
		H = np.matmul(G,H) #al final de la iteracion: H=R
	Q = Q.transpose()
	return [Q,H]

#Dada una matriz de hessenberg H (m+1)*m y un vector c m+1, retorna el vector solucion al PMCL mas la descomposicion QR fina de H 
def pmcl(H,c):
	QR = hessenbergQR(H)
	colsR = QR[1].shape[1]
	R1 = QR[1][0:colsR,0:colsR]	
	X = np.zeros(colsR)
	c = np.matmul(QR[0].transpose(),c)
	for i in range(colsR-1,-1,-1):
		tmp = c[i]
		for j in range(colsR-1, i, -1):
			tmp -= X[j]*R1[i,j]
		X[i] = tmp/R1[i,i]
	return [X,QR]

#Calcula la proyeccion de v segun el vector u: proj_u(v) = (<v,u>/<u,u>)*u
def proy(v,u):
	if not np.count_nonzero(u):
		return np.zeros(u.shape[1])
	return (np.dot(v,u)/np.dot(u,u))*u
#Retorna el vector v normalizado (v != 0)
def normalizar(v):
	return v/np.sqrt(v.dot(v))
#Dada una matrix A m*n, devuelve Q m*n y R n*n tq A=QR. Q vectores ortonormales y R triangular superior
def gsc(A):
	Q = A.copy()
	n = A.shape[1]
	Q[:,0] = normalizar(Q[:,0])
	for i in range(1,n):
		Qi = Q[:,i]
		for j in range(0,i):
			Qj = Q[:,j]
			t = Qi.dot(Qj)
			Qi = Qi-t*Qj
		Q[:,i] = normalizar(Qi)
	return [Q,np.matmul(Q.transpose(),A)]

#Gram-Schmidt modificado. Dada una matrix A m*n, devuelve Q m*n y R n*n tq A=QR. Q vectores ortonormales y R triangular superior
def gsm(A):
	n = A.shape[1]
	Q = A.copy()
	for j in range(0,n):
		Q[:,j] = Q[:,j]/np.linalg.norm(Q[:,j])
		for k in range(j+1,n):
			Q[:,k] = Q[:,k] - np.dot(Q[:,j],Q[:,k])*Q[:,j]
	return [Q,np.matmul(Q.transpose(),A)]
#Algoritmo adaptado de las notas de Tom Trogdon, Universidad de Mitchigan. Disponible en: math.uci.edu/~ttrodgon (Lecture 23)

#Algoritmo de Arnoldi. Dada una matriz A, vector r y entero n. Devuelve una base ortonormal del subespacio de Krylov de dimension n (K), y una matriz H de Hessenberg.
def arnoldi(A,r,n):
	m = A.shape[0]
	K = np.zeros((m,n))
	H = np.zeros((n+1,n))
	K[:,0] = r/np.linalg.norm(r)
	for j in range(0,n):
		wj = np.matmul(A,K[:,j])
		for i in range(0,j+1):
			H[i,j] = np.dot(K[:,i],wj) 
			wj = wj - H[i,j]*K[:,i]
		H[j+1,j] = np.sqrt(wj.dot(wj))
		if H[j+1,j] == 0:
			break;
		if j != n-1:
			K[:,j+1] = wj/H[j+1,j]
	return [K,H]
#Algoritmo adaptado de Y. Saad, Iterative Methods for Sparse Linear Systems. pg.162 

#Dada una matriz A, vector b, vector sig y matrices Q_m y H_m del paso anterior retorna Q_m+1,H_m+1 donde Q y H son las matrices del algoritmo de Arnoldi. Es decir, retorna la siguiente iteracion del algoritmo
def arnold(A,Q,H,b,m,sig):
	if m==1:
		Q[:,0] = (b/np.linalg.norm(b))
		wj = np.matmul(A,Q[:,0])
		H[0,0] = np.dot(Q[:,0],wj)
		wj = wj-H[0,0]*Q[:,0]
		H[1,0] = np.linalg.norm(wj)
		sig = wj/H[1,0]
		return [Q,H,sig]	
	Q[:,m-1] = sig
	wj = np.matmul(A,sig)
	for i in range(0,m):
		H[i,m-1] = np.dot(wj,Q[:,i])
		wj = wj - H[i,m-1]*Q[:,i]
	H[m,m-1] = np.linalg.norm(wj)
	if H[m-1,m-2] == 0:
		print("ERROR! Es A linealmente independiente?")
		return [Q.fill(666),H.fill(666)]
	sig = wj/H[m,m-1]
	return [Q,H,sig]

#Algoritmo GMRES. Dado un vector x0 arbitrario, retorna la solucion del problema Ax=b. Donde el resto de la solucion tiene la norma relativa menor a la tolerancia O se llega al nro maximo de iteraciones
def gmres(A,x0,b,tol,maxIter):
	ITER = maxIter
	r0 = b-np.matmul(A,x0)[:,0]
	m = 1
	n = A.shape[1]
	Mmax = min(ITER,n)
	rmabs=np.linalg.norm(r0)
	QH = [np.zeros((n,n)),np.zeros((n+1,n))]		
	sig = np.zeros((A.shape[0],1))
	nrevs = [1]
	nrev = 1
	while(m<=Mmax and nrev>tol):
		QH = arnold(A,QH[0],QH[1],b,m,sig)
		sig = QH[2]
		c = np.zeros((m+1,1))
		c[0,0] = np.linalg.norm(r0)
		YmQR_H = pmcl(QH[1][0:m+1,0:m],c)
		ym = YmQR_H[0]
		rmabs = abs(np.matmul(YmQR_H[1][0].transpose(),c))[m,0]
		m=m+1
		nrev = rmabs/np.linalg.norm(r0)
		nrevs.append(nrev)
	m=m-1
	return [x0[:,0]+np.matmul(QH[0][:,0:m],ym),QH[0],nrevs]

#Test con un vector aleatorio
def testEje1():
	X = np.array([random.randint(-100,100),random.randint(-100,100)])
	P = givensParam(X)
	G = np.array([[P[0],P[1]],[-P[1],P[0]]])
	r = np.dot(G.transpose(),X)
	with np.printoptions(precision=8, suppress=True):
		print("\nVector X aleatorio: ",X)
		print("\nRotacion de Givens G:\n",G)
		print("\nG^t*X=\n",np.matmul(G.transpose(),X))
	
def testEje2():
	H = np.array([[2,0,1,5],
		     [-3,2,-6,4],
	             [0,4,7,1],
		     [0,0,-2,0],
	             [0,0,0,6]])
	QR = hessenbergQR(H)
	with np.printoptions(precision=8, suppress=True):
		print("H:\n",H,"\nQ:\n",QR[0],"\nR:\n",QR[1],"\n")
		res = np.matmul(QR[0],QR[1])
		print("\nQ*R:\n",res)
		print("\nQ*Q^t:\n",np.matmul(QR[0],QR[0].transpose()),"\nQ^t*Q:\n",np.matmul(QR[0].transpose(),QR[0]),"\n")

def testEje3():
	H = np.array([[2,0,1,5],
		     [-3,2,-6,4],
	            [0,4,7,1],
		     [0,0,-2,0],
	             [0,0,0,6]])
	c = np.array([[3],
			[0],
			[0],
			[0],
			[0]])
	X = pmcl(H,c)
	with np.printoptions(precision=8, suppress=True):
		print("H:\n",H)
		print("c:\n",c)
		print("\nX^t:\n",X[0],"\n")
		print("(H*X)^t:\n",np.matmul(H,X[0].transpose()))
def testEje4():
	H = hilbert(7)
	M = magic(7)
	QRH = gsc(H)
	QRM = gsc(M)
	with np.printoptions(precision=16, suppress=True):
		print("\nMatriz de Hilbert H:\n",H,"\nMatriz magica M:\n",M,"\nPrueba de GSC para H:\n")
		print("Q=\n",QRH[0],"\nR=\n",QRH[1],"\nQ*R=\n",np.matmul(QRH[0],QRH[1]),"\nQ^t*Q=\n",np.matmul(QRH[0].transpose(),QRH[0]))
		print("\nPrueba de GSC para M:\nQ=\n",QRM[0],"\nR=\n",QRM[1],"\nQ*R=\n",np.matmul(QRM[0],QRM[1]),"\nQ^t*Q=\n",np.matmul(QRM[0].transpose(),QRM[0]))	
		QRH = gsm(H)
		QRM = gsm(M)
		print("Prueba de GSM para H:\nQ=\n",QRH[0],"\nR=\n",QRH[1],"\nQ*R=\n",np.matmul(QRH[0],QRH[1]),"\nQ^t*Q=\n",np.matmul(QRH[0].transpose(),QRH[0]))
		print("\nPrueba de GSM para M:\nQ=\n",QRM[0],"\nR=\n",QRM[1],"\nQ*R=\n",np.matmul(QRM[0],QRM[1]),"\nQ^t*Q=\n",np.matmul(QRM[0].transpose(),QRM[0]))	
		
def testEje5():
	A = np.array([[5,1,0,-8],[0,-2,3,1],[-9,13,1,6],[1,0,5,3]])
	r = np.array([7,-2,0,4])
	m = 4 
	KH = arnoldi(A,r,m)
	with np.printoptions(precision=8, suppress=True):
		print("\nA=\n",A,"\nr=",r)
		print("Q:\n",KH[0],"\n~H:\n",KH[1])
		print("\nQ*(H)*Q^t=\n",np.matmul(KH[0],np.matmul(KH[1][:-1,:],KH[0].transpose())))
		print("Donde K es una base ortonormal del subespacio de Krylov de dimension m=",m)
def testEje7():
	n = input("Inserte la dimensión de la matriz (n): ")
	d = input("Inserte la densidad (d): ")
	print("Distribucion de la matriz esparsa: N(u,r/sqrt(n*d))")
	u = input("Inserte u: ")
	r = input("Inserte r: ")

	n = int(n)
	d = float(d.replace(',','.'))
	u = float(u.replace(',','.'))
	r = float(r.replace(',','.'))

	def normaldist(n):
		return np.random.normal(0,1,n)
	S = sparse.random(n,n,density=d, data_rvs=normaldist)
	A = u*np.identity(n)+((r)/np.sqrt(n*d))*np.array(S.toarray())
	X0 = np.zeros((n,1))
	b_s = sparse.random(n,1,density=d,data_rvs=normaldist)
	b = np.array(b_s.toarray())[:,0]	
	tol = 1E-6
	
	EIG = np.linalg.eig(A)
	X = [x.real for x in EIG[0]]
	Y = [x.imag for x in EIG[0]]
	plt.scatter(X,Y, color = 'blue')
	plt.title("Gráfica de los VAPS de A: λ = a+bi")
	plt.xlabel('a')
	plt.ylabel('b')
	plt.savefig(os.path.dirname(__file__)+"vaps.png", dpi=500)
	plt.clf()
	MAXITER = 30	
	RES = gmres(A,X0,b,tol,MAXITER)	
	k = np.linalg.cond(EIG[1])
	m = np.linspace(0,len(RES[2]),100)
	cot = k*((r/u)**m) #r -> radio u->centro 
	X = [x for x in range(0,len(RES[2]))]	
	plt.plot(m,cot,'red',label="Cota teórica")
	plt.plot(X,RES[2],color = 'blue',label="Residuo")
	plt.legend()
	plt.title("Grafica de la norma del residuo relativo en función del paso iterativo m")
	plt.xlabel('m')
	plt.ylabel('||rm||')
	plt.yscale("log")
	plt.savefig(os.path.dirname(__file__)+"residuo.png", dpi=500)
	plt.clf()
	
	solv = np.linalg.solve(A,b)	
	f = open(os.path.dirname(__file__)+"resultados.txt","w")
	f.write("---Compracion del Vector Solucion X entre GMRES y numpy.linalg.solve---\nGMRES		| numpy.linalg.solve\n")
	f.close()
	with open(os.path.dirname(__file__)+"resultados.txt","a") as f:
		for x in range(0,len(RES[0])):
			f.write(str(RES[0][x])+" | "+str(solv[x])+"\n")

	print("Se ha generado en el directorio donde se ejecuto el programa:\nvaps.png con el grafico de los valores propios de A.\nresiduo.png con la grafica del residuo relativo en funcion de m\nresultados.txt",
	"con la comparacion de los vectores solucion entre nuestro algoritmo y numpy.linalg.solve")			
def tiempoGMRES():
	print("ADVERTENCIA: Se va a ejecutar el algoritmo GMRES y la funcion numpy.linalg.solve 3xN veces. Esto puede llevar un tiempo considerable dependiendo de su ordenador.\n")
	top = input("Inserte dimension maxima a iterar (N>10):")
	top = int(top)
	tol = 1E-6
	den = 0.01
	u = 2
	r = 0.5 
	MAXITER = 30
	def normaldist(n):
		return np.random.normal(0,1,n)
	gmresTokei = []
	linalgTokei = []
	for n in range(10,top+1):
		S = sparse.random(n,n,density=den, data_rvs=normaldist)
		A = u*np.identity(n)+((r)/np.sqrt(n*den))*np.array(S.toarray())
		X0 = np.zeros((n,1))
		b_s = sparse.random(n,1,density=den,data_rvs=normaldist)
		b = np.array(b_s.toarray())[:,0]	
		d = []
		g = []
		for k in range(0,3):
			start = time.time()
			RES = gmres(A,X0,b,tol,MAXITER)
			end = time.time()
			d.append(end-start)
			start = time.time()
			solv = np.linalg.solve(A,b)
			end = time.time()
			g.append(end-start)
		avg = (d[0]+d[1]+d[2])/3.0
		gmresTokei.append(avg)
		avg = (g[0]+g[1]+g[2])/3.0
		linalgTokei.append(avg)	
		print("Resolviendo sistema de dimensión",n,end='\r')
	N = [n for n in range(10,top+1)]

	plt.plot(N,gmresTokei,color='red')
	plt.title("Gráfica del tiempo de ejecución de GMRES en función de la dimensión\n de A")
	plt.xlabel('n')
	plt.ylabel('t (s)')
	plt.savefig(os.path.dirname(__file__)+"tiempo_gmres.png",dpi=500)
	plt.clf()
	
	plt.plot(N,linalgTokei,color='red')
	plt.title("Gráfica del tiempo de ejecución de numpy.linalg.solve en función de\n la dimensión de A")
	plt.xlabel('n')
	plt.ylabel('t (s)')
	plt.savefig(os.path.dirname(__file__)+"tiempo_linalg.png",dpi=500)
	plt.clf()
	
	print("Se han generado las siguientes gráficas de tiempo:\n tiempo_linalg.png con la gráfica de tiempo de ejecución en función del tamaño de la matriz\ntiempo_gmres.png con la gráfica del tiempo de ejecucón",
	"en función del tamaño de la matriz. Amabas imagenes se han guardado en el directorio donde se ejecutó el programa")

def GMREStime():
	print("ADVERTENCIA: Se va a ejecutar el algoritmo GMRES 3xN veces. Esto puede llevar un tiempo considerable dependiendo de su ordenador.\n")
	top = input("Inserte la dimensión maxima de A (N>10):")
	top = int(top)
	tol = 1E-6
	den = 0.01
	u = 2
	r = 0.5
	MAXITER = 30
	def normaldist(n):
		return np.random.normal(0,1,n)
	gmresTokei = []
	linalgTokei = []
	for n in range(10,top+1):
		S = sparse.random(n,n,density=den, data_rvs=normaldist)
		A = u*np.identity(n)+((r)/np.sqrt(n*den))*np.array(S.toarray())
		X0 = np.zeros((n,1))
		b_s = sparse.random(n,1,density=den,data_rvs=normaldist)
		b = np.array(b_s.toarray())[:,0]	
		d = []
		g = []
		for k in range(0,3):
			start = time.time()
			RES = gmres(A,X0,b,tol,MAXITER)
			end = time.time()
			d.append(end-start)
		avg = (d[0]+d[1]+d[2])/3.0
		gmresTokei.append(avg)
		print("Resolviendo sistema de dimension",n,end='\r')
	N = [n for n in range(10,top+1)]
	plt.plot(N,gmresTokei,color='red')
	plt.title("Gráfica del tiempo de ejecución de GMRES en función de la dimensión\n de A")
	plt.xlabel('n')
	plt.ylabel('t (s)')
	plt.savefig(os.path.dirname(__file__)+"tiempo_gmres2.png",dpi=600)
	plt.clf()
	print("Se han generado las siguientes gráficas de tiempo:tiempo_gmres2.png con la grafica del tiempo de ejecución",
	"en función del tamaño de la matriz. Amabas imagenes se han guardado en el directorio donde se ejecutó el programa")


def main():
	print("IMPLEMENTACION OBLIGATORIO 1 - GMRES\n")
	print("Seleccione el ejercicio a ejecutar (1-8):\n",
	"1_ Rotaciones de givens\n 2_ QR Matrix de Hessenberg\n 3_ PMCL con QR fino\n 4_ Gram-Schmidt\n 5_ Base Ortonormal de Subespacio de Krylov (Arnoldi)\n 6_ GMRES\n 7_ Tiempo GMRES vs numpy.linalg.solve\n 8_ Tiempo GMRES")
	eje = input(">")
	options = {1: testEje1, 2: testEje2, 3: testEje3, 4: testEje4, 5: testEje5, 6: testEje7, 7: tiempoGMRES, 8: GMREStime}
	options[int(eje)]()

if __name__ == '__main__':
	main()
	#Extra()

