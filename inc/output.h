#ifndef OUTPUT_H
#define OUTPUT_H

// General C++ headers.
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <omp.h>

// Eigen.
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

// VTK.
#include "vtkAMRBox.h"
#include "vtkAMRUtilities.h"
#include "vtkCell.h"
#include "vtkCellData.h"
#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkNew.h"
#include "vtkOverlappingAMR.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkUniformGrid.h"
#include "vtkImageData.h"
#include "vtkXMLImageDataWriter.h"
#include "vtkXMLUniformGridAMRWriter.h"
#include "vtkCellDataToPointData.h"
#include "vtkContourFilter.h"
#include "vtkActor.h"
#include "vtkImageActor.h"
#include "vtkImageProperty.h"
#include "vtkImageCast.h"
#include "vtkLookupTable.h"
#include "vtkColorTransferFunction.h"
#include "vtkImageMapToColors.h"
#include "vtkImageMapper3D.h"
#include "vtkDataSetMapper.h"
#include "vtkCamera.h"
#include "vtkGraphicsFactory.h"
#include "vtkNamedColors.h"
#include "vtkPNGWriter.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkWindowToImageFilter.h"

#endif
