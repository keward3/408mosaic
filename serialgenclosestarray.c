int* genclosestarray(int* compvals, int* sectvals, int numcompimages, int numsections)
{
    int* closestarray = malloc(numsections * sizeof(int));
    for(int i = 0; i < numsections; i++){
        int closestval = 0;
        int closestlocation = 0;
        for(int k = 0; k < 12; k++){
            closestval += (sectvals[i*12 + k] - compvals[k]) * (sectvals[i*12 + k] - compvals[k]);
        }
        for(int j = 1; j < numcompimages; j++){
            int closestcompare = 0;
            for(int k = 0; k < 12; k++){
                closestcompare += (sectvals[i*12 + k] - compvals[j*12 + k]) * (sectvals[i*12 + k] - compvals[j*12 + k]);
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
