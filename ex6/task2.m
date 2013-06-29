load('ex6data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);
arrayC = [30000;10;3;1;0.3;0.1;0.03;0.01];
arrayC = arrayC(:);
arraySigma = [0.01;0.03;0.1;0.3;1;3;10;30];
arraySigma = arraySigma(:);


%Train the SVM
% for i = 1:size(arrayC)
	% for j = 1:size(arraySigma)	
		% C = arrayC(i)
		% sigma = arraySigma(j)
		% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		% visualizeBoundary(X, y, model)
		% fprintf('Program paused. Press enter to continue.\n');
		% pause
	% endfor
% endfor

C = 10;
sigma = 0.01;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);