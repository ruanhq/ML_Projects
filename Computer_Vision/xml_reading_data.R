###Read in the xml data and extract all the annotated_bounding_box corresponding to the matrix:
#So we can quantitatively evaluate the performance.
fileList<-list.files(path=".")
for(i in 1:length(fileList)){
  current_xml<-xmlToList(fileList[i])
  image_index<-i+1590
  ##
  #If the there are more than 1 bounding box annotated in the image, we just set 
  if(length(current_xml)==7){
    bounding_box_information<-current_xml$object$bndbox
    xmin<-as.numeric(bounding_box_information$xmin)
    ymin<-as.numeric(bounding_box_information$ymin)
    xmax<-as.numeric(bounding_box_information$xmax)
    ymax<-as.numeric(bounding_box_information$ymax)
    bounding_information<-matrix(c(xmin,ymin,xmax,ymax),1,4,byrow=TRUE)
    rownames(bounding_information)<-NULL
    colnames(bounding_information)<-NULL
    write.table(bounding_information,file=paste(image_index,'bbox.txt'))
  }
  else{
    number_of_bounding_box<-length(current_xml)-6
    bounding_information<-matrix(0,number_of_bounding_box,4)
    for(j in 1:number_of_bounding_box){
      bbx<-current_xml[6+j]$object$bndbox
      bounding_information[j,1]<-as.numeric(bbx$xmin)
      bounding_information[j,2]<-as.numeric(bbx$ymin)
      bounding_information[j,3]<-as.numeric(bbx$xmax)
      bounding_information[j,4]<-as.numeric(bbx$ymax)
    }
    rownames(bounding_information)<-NULL
    colnames(bounding_information)<-NULL
    write.table(bounding_information,file=paste(image_index,'bbox.txt'))
  }
}
