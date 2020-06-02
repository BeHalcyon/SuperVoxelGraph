#include "VolumeToLabel.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	VolumeToLabel w;
	w.show();
	w.setWindowTitle("Labeling Voxel Tool");
	return a.exec();
}
