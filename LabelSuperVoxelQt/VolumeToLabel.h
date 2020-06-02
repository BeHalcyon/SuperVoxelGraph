#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_VolumeToLabel.h"
#include "../SLIC3DSuperVoxel/SourceVolume.h"

#include <fstream>
#include <iostream>
#include "ParameterControlWidget.h"
#include <QDockWidget>

#include "../SLIC3DSuperVoxel/json_struct.h"
#include "SliceView.h"

class VolumeToLabel : public QMainWindow
{
	Q_OBJECT

public:
	VolumeToLabel(QWidget* parent = Q_NULLPTR);

	void setConnectionState();
	bool getDrawedState() const { return is_drawed; }

public slots:
	void slot_ImportVifoFile();
	void slot_ExportNetAndNodeFile();
	
	//Extension 20200602
	void slot_ImportJsonFile();
	void slot_ExportLabeledClusterCSV();
	void loadLabelVolume();
	void loadVolume();
private:
	Ui::VolumeToLabelClass	ui;
	std::string				infoFileName = "E:\\atmosphere\\timestep21_float\\_SPEEDf21.vifo";
	int						data_number;
	std::string				datatype;
	hxy::my_int3			dimension;
	hxy::my_double3			space;
	vector<string>			file_list;
	vector<unsigned char>	volume_data;
	std::string				file_path;

	// Extenstion 20200602
	std::string				json_file = "../x64/Release/workspace/spheres_supervoxel.json";
	ConfigureJSONStruct		configure_json;
	vector<double>			volume_label_data;		//int type

	ParameterControlWidget* parameter_control_widget;
	QDockWidget* parameter_dock;
	SliceView* slice_view;
	int						plane_mode = 0;	//0 for xy-plane, 1 for yz-plane, 2 for xz-plane;
	int						slice_id = 0;
	bool					is_drawed = false;
};
