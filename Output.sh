command clear
printf "Naive Bayes ALgorithm Analysis with different Distributed Computing Modules\n\n"
printf "********\n\n"
printf "Method 1: Plain Naive Bayes\nMethod 2: Naive Bayes with MPI\nMethod 3: Naive Bayes with Multiprocessing and MultiThreading\n\n"
printf "********\n\n"
echo "Normal Naive Bayes Program: "
command python3 PlainNaiveBayes.py
echo ""
printf "********\n\n"
echo "Naive Bayes with MPI: "
command mpiexec -n 4 python3 MPINaiveBayes.py
echo ""
printf "********\n\n"
echo "Naive Bayes with Multiprocessing and Multithreadiung: "
command python3 DistributedNaiveBayes.py
printf "\n\n"