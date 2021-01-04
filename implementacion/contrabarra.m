function T = contrabarra (n)
  T = [];
  N = 1:n;
  for n = 1:n;
    A = 2*eye(n) + (0.5/sqrt(0.01*n))*sprandn(n,n,0.01);
    b = sprand(n,1,0.01);
    tic();
    A\b;
    t1 = toc();
    tic();
    A\b;
    t2 =toc();
    tic();
    A\b;
    t3 = toc();
    T(end+1) = (t1+t2+t3)/3;
    printf("Resolviendo sistema n=%d \r",n);
  endfor
  plot(N,T,'m');
  xlabel("n")
  ylabel("t (s)")
  title("Gráfica del tiempo de ejecución del operador 'contrabarra' de Octave en función de la dimensión de A");
endfunction

