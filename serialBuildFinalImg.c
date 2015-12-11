unsigned int* serialBuildFinalImg(int* closestFit, Uint8* components, int sectionsPerRow, int pixelsPerSection)
{
    int* finalImage = malloc(FINAL_HEIGHT * FINAL_WIDTH * 4 * sizeof(Uint8));
    for(int i = 0; i < FINAL_HEIGHT * FINAL_WIDTH; i++){
        int xIndex = i % FINAL_WIDTH;
        int yIndex = i / FINAL_WIDTH;

        int sectionIndex = (xIndex / COMP_WIDTH) + (yIndex / COMP_HEIGHT) * sectionsPerRow;
        int closest = closestFit[sectionIndex];

        int sectionIndexX = xIndex % COMP_WIDTH;
        int sectionIndexY = yIndex % COMP_HEIGHT;

        finalImage[i * 4] = components[closest * pixelsPerSection + sectionIndexY * COMP_WIDTH + sectionIndexX];
        finalImage[i * 4 + 1] = components[closest * pixelsPerSection + sectionIndexY * COMP_WIDTH + sectionIndexX + 1];
        finalImage[i * 4 + 2] = components[closest * pixelsPerSection + sectionIndexY * COMP_WIDTH + sectionIndexX + 2];
        finalImage[i * 4 + 3] = 0xff;
    }
    return finalImage;
}
