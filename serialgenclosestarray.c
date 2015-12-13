int* genclosestarray(int* compvals, int* sectvals, int numcompimages, int numsections)
{
    int* closestarray = (int*)malloc(numsections * sizeof(int));
    for(int i = 0; i < numsections; i++){
        int closestval = 0;
        int closestlocation = 0;
        for(int k = 0; k < 4; k++){
            closestval += (sectvals[i*4 + k] - compvals[k]) * (sectvals[i*4 + k] - compvals[k]);
        }
        for(int j = 1; j < numcompimages; j++){
            int closestcompare = 0;
            for(int k = 0; k < 4; k++){
                closestcompare += (sectvals[i*4 + k] - compvals[j*4 + k]) * (sectvals[i*4 + k] - compvals[j*4 + k]);
            }
            if(closestcompare < closestval){
                closestval = closestcompare;
                closestlocation = j;
            }
        }
        closestarray[i] = closestlocation;
    }
    return closestarray;
}
