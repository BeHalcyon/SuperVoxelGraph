#include "VolumeToLabel.h"
#include "../SLIC3DSuperVoxel/SourceVolume.cpp"
#include <QFileDialog>
#include <QDebug>
#include <QGraphicsPathItem>
#include <set>

void readInfoFile(const std::string& infoFileName, int& data_number, std::string& datatype, hxy::my_int3& dimension, hxy::my_double3& space,
                  std::vector<std::string>& file_list)
{
	file_list.clear();

	ifstream inforFile(infoFileName);

	inforFile >> data_number;
	inforFile >> datatype;
	inforFile >> dimension.x >> dimension.y >> dimension.z;
	//Debug 20190520 增加sapce接口
	inforFile >> space.x >> space.y >> space.z;
	const string filePath = infoFileName.substr(0, infoFileName.find_last_of('/') == -1 ?
		infoFileName.find_last_of('\\') + 1 : infoFileName.find_last_of('/') + 1);
	std::cout << (filePath.c_str()) << std::endl;
	for (auto i = 0; i < data_number; i++)
	{
		string rawFileName;
		inforFile >> rawFileName;
		string volumePath = filePath + rawFileName;
		file_list.push_back(volumePath);
	}
	std::cout << "Info file name : \t\t" << infoFileName.c_str() << std::endl;
	std::cout << "Volume number : \t\t" << data_number << std::endl;
	std::cout << "data type : \t\t\t" << datatype.c_str() << std::endl;
	std::cout << "Volume dimension : \t\t" << "[" << dimension.x << "," << dimension.y << "," << dimension.z << "]" << std::endl;
	std::cout << "Space dimension : \t\t" << "[" << space.x << "," << space.y << "," << space.z << "]" << std::endl;
	for (auto i = 0; i < data_number; i++)
	{
		std::cout << "Volume " << i << " name : \t\t" << file_list[i].c_str() << std::endl;
	}

	std::cout << "Info file has been loaded successfully." << std::endl;
}


VolumeToLabel::VolumeToLabel(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setWindowState(windowState() ^ Qt::WindowMaximized);

	parameter_control_widget = new ParameterControlWidget(this);
	parameter_dock = new QDockWidget(tr("Parameter Setting"), this);

	parameter_dock->setWidget(parameter_control_widget);
	addDockWidget(Qt::LeftDockWidgetArea, parameter_dock);

	slice_view = new SliceView(this);
	setCentralWidget(slice_view);


	setConnectionState();
}

void VolumeToLabel::setConnectionState()
{
	connect(ui.action_vifo, &QAction::triggered, this, &VolumeToLabel::slot_ImportVifoFile);
	//Extension 20200602
	connect(ui.action_json, &QAction::triggered, this, &VolumeToLabel::slot_ImportJsonFile);
	connect(ui.action_csv, &QAction::triggered, this, &VolumeToLabel::slot_ExportLabeledClusterCSV);
	connect(parameter_control_widget->ui.comboBox_Slice_Direction, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index)
		{
			if (index == 0)
				parameter_control_widget->ui.spinBox_Slice_index->setMaximum(dimension.z);
			else if (index == 1)
				parameter_control_widget->ui.spinBox_Slice_index->setMaximum(dimension.x);
			else
				parameter_control_widget->ui.spinBox_Slice_index->setMaximum(dimension.y);

			slice_id = 0;

			plane_mode = index;
			if (is_drawed)
			{
				slice_view->updateImage(volume_data, dimension, plane_mode, slice_id);
			}
		});
	connect(parameter_control_widget->ui.spinBox_Slice_index, QOverload<int>::of(&QSpinBox::valueChanged), [this](int value)
		{
			slice_id = value;
			if (is_drawed)
			{
				//std::cout << "test" << std::endl;
				slice_view->updateImage(volume_data, dimension, plane_mode, slice_id);
			}
		});

	connect(slice_view, &SliceView::signal_updateSliceId, [this](int offset)
		{
			if (offset == 1)
			{
				if (parameter_control_widget->ui.spinBox_Slice_index->maximum() > slice_id + 1)
					parameter_control_widget->ui.spinBox_Slice_index->setValue(slice_id + 1);
			}
			else if (offset == -1)
			{
				if (0 <= slice_id - 1)
					parameter_control_widget->ui.spinBox_Slice_index->setValue(slice_id - 1);
			}
		});

	connect(parameter_control_widget->ui.pushButton_add_label, &QPushButton::clicked, [this]()
		{
			const auto label_name = parameter_control_widget->ui.line_edit_new_label_name->text();

			auto combo_list = parameter_control_widget->ui.comboBox_label_name_list;
			for (auto i = 0; i < combo_list->count(); i++)
			{
				if (label_name == combo_list->itemText(i)) return;
			}
			combo_list->addItem(label_name);
			slice_view->createNewPathItemArray(label_name);
		});

	connect(parameter_control_widget->ui.pushButton_select_label, &QPushButton::clicked, [this]()
		{
			const auto label_name = parameter_control_widget->ui.comboBox_label_name_list->currentText();
			slice_view->setLabel(label_name);
		});
	connect(slice_view, &SliceView::signal_updateLableNumber, [this](const QVector<QVector<QVector<QVector<QGraphicsPathItem*>>>>& vector)
		{

			auto combo_list = parameter_control_widget->ui.comboBox_label_name_list;
			for (auto i = 0; i < combo_list->count(); i++)
			{
				const auto label_name = combo_list->itemText(i);
				const auto label_id = slice_view->getLabelId(label_name);
				if (label_id == -1)
				{
					qDebug() << "Error: label id is -1";
					return;
				}
				parameter_control_widget->ui.tableWidget->setItem(i, 0, new QTableWidgetItem(label_name));

				int cnt = 0;

				for (const auto& j : vector[label_id])
				{
					for (const auto& k : j)
					{
						for (auto buf : k)
						{
							cnt += buf->path().elementCount();
						}
					}
				}
				parameter_control_widget->ui.tableWidget->setItem(i, 1, new QTableWidgetItem(QString("%1").arg(cnt)));
			}
		});

	connect(ui.action_net, &QAction::triggered, this, &VolumeToLabel::slot_ExportNetAndNodeFile);

	connect(ui.action_node, &QAction::triggered, [this]() {

		QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"),
			"./workspave/labels.node",
			tr("label node (*.node)"));
		if (fileName.isEmpty())
			return;

		/***
		 * 以二进制形式存储非重复label文件，文件格式："
		 * label:1
		 * label:2
		 * label:3
		 * "
		 */
		auto fo = fopen(fileName.toStdString().c_str(), "wb");
		auto combo = parameter_control_widget->ui.comboBox_label_name_list;
		for (auto i = 0; i < combo->count(); i++)
		{
			const auto label_name = combo->itemText(i);
			fprintf(fo, "%s\n", label_name.toStdString().c_str());
		}
		fclose(fo);
		});

}

void VolumeToLabel::slot_ExportNetAndNodeFile()
{

	QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"),
		"./workspave/lw.net",
		tr("label-word net (*.net)"));
	if (fileName.isEmpty())
		return;

	//全局到局部转换,过滤外界点,平面到立体
	auto label_array = slice_view->tranPathListToVolumeIndex();


	//获取window size
	int window_size = 3;
	const auto sz = dimension.x * dimension.y * dimension.z;
	QVector<QVector<int>> context_label(label_array.size());
	for (auto i = 0; i < context_label.size(); i++) context_label[i].resize(256);

	if (window_size == 3)
	{
		const int dx26[26] = { -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1 };
		const int dy26[26] = { -1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1 };
		const int dz26[26] = { -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };


		for (auto i = 0; i < label_array.size(); i++)
		{
			for (auto index = 0; index < label_array[i].size(); index++)
			{
				auto center_index = label_array[i][index];

				const int oz = center_index / (dimension.x * dimension.y);
				const int ox = center_index % dimension.x;
				const int oy = (center_index % (dimension.x * dimension.y)) / dimension.x;

				context_label[i][volume_data[center_index]]++;

				for (auto p = 0; p < 26; p++)
				{
					int nx = ox + dx26[p];//new x
					int ny = oy + dy26[p];//new y
					int nz = oz + dz26[p];//new z

					if (nx >= 0 && nx < dimension.x && ny >= 0 && ny < dimension.y && nz >= 0 && nz < dimension.z)
					{
						int nind = nz * dimension.x * dimension.y + ny * dimension.x + nx;
						context_label[i][volume_data[nind]]++;
					}
				}
			}
		}
	}
	else if (window_size == 1)
	{
		const int dx6[6] = { -1,  1,  0,  0,  0,  0 };
		const int dy6[6] = { 0,  0, -1,  1,  0,  0 };
		const int dz6[6] = { 0,  0,  0,  0, -1,  1 };


		for (auto i = 0; i < label_array.size(); i++)
		{
			for (auto index = 0; index < label_array[i].size(); index++)
			{
				auto center_index = label_array[i][index];

				const int oz = center_index / (dimension.x * dimension.y);
				const int ox = center_index % dimension.x;
				const int oy = (center_index % (dimension.x * dimension.y)) / dimension.x;

				context_label[i][volume_data[center_index]]++;

				for (auto p = 0; p < 6; p++)
				{
					int nx = ox + dx6[p];//new x
					int ny = oy + dy6[p];//new y
					int nz = oz + dz6[p];//new z

					if (nx >= 0 && nx < dimension.x && ny >= 0 && ny < dimension.y && nz >= 0 && nz < dimension.z)
					{
						int nind = nz * dimension.x * dimension.y + ny * dimension.x + nx;
						context_label[i][volume_data[nind]]++;
					}
				}
			}
		}
	}

	//保存为文件

	auto fo = fopen(fileName.toStdString().c_str(), "wb");
	//以label为数量循环
	auto combo = parameter_control_widget->ui.comboBox_label_name_list;

	for (int m = 0; m < context_label.size(); m++)
	{
		const auto label_name = combo->itemText(m);
		for (int n = 0; n < context_label[m].size(); n++)
		{
			if (context_label[m][n] > 0)
			{
				fprintf(fo, "%s %d %d l\n", label_name.toStdString().c_str(), n, context_label[m][n]);
			}
		}

	}
	fclose(fo);
}


void VolumeToLabel::slot_ExportLabeledClusterCSV()
{
	//全局到局部转换,过滤外界点,平面到立体
	auto label_array = slice_view->tranPathListToVolumeIndex();

	std::string file_prefix = configure_json.data_path.file_prefix;

	//vector<set<int>> label_id_set_vector;
	//for (auto i = 0; i < label_array.size(); i++)
	//{

	//	set<int> label_id_set;
	//	for (auto index = 0; index < label_array[i].size(); index++)
	//	{
	//		label_id_set.insert(static_cast<int>(volume_label_data[label_array[i][index]]));
	//	}

	//	label_id_set_vector.push_back(label_id_set);
	//	
	//	
	//}
	std::string voxels_labeled_file_name = configure_json.file_name.labeled_voxel_file;
	// Save the data to csv file
	std::string csv_file_name = file_prefix + voxels_labeled_file_name;
	ofstream writer(csv_file_name);

	writer << "VoxelPos,LabelID" << endl;
	for (auto i = 0; i < label_array.size(); i++)
	{
		set<int> label_id_set;
		for (auto index = 0; index < label_array[i].size(); index++)
		{
			writer << label_array[i][index] << "," << i << endl;
		}
	}

	//int cluster_id = 0;
	//for(auto & labeled_set: label_id_set_vector)
	//{
	//	for (auto value : labeled_set)
	//	{
	//		writer << value << "," << cluster_id << endl;
	//	}
	//	cluster_id++;
	//}
	writer.close();
}


// Extension 20200602
void VolumeToLabel::loadLabelVolume()
{
	std::string volume_file_path = configure_json.data_path.file_prefix;
	std::string volume_file_name = configure_json.file_name.label_file;
	SourceVolume source_volume({ volume_file_path + volume_file_name }, dimension.x, dimension.y, dimension.z, "int");

	source_volume.loadVolume();
	volume_label_data = *source_volume.getOriginVolume(0);
}


// Extension 20200602
void VolumeToLabel::slot_ImportJsonFile()
{
	QString file_name = QFileDialog::getOpenFileName(this, "Open json File", "D:/project/science_project/SLIC3DSuperVoxel/x64/Release/workspace/",
		tr("JSON (*.json )"));
	if (file_name.isEmpty())
		return;
	json_file = file_name.toStdString();
	
	try
	{
		std::ifstream input_file(json_file);
		input_file >> configure_json;

		infoFileName = configure_json.data_path.vifo_file;
		readInfoFile(infoFileName, data_number, datatype, dimension, space, file_list);
		file_path = file_list[0].substr(0, file_list[0].find_last_of('.'));

		loadVolume();
		loadLabelVolume();
	}
	catch (std::exception& e)
	{
		vm::println("{}", e.what());
	}
}


void VolumeToLabel::slot_ImportVifoFile()
{
	QString fileName = QFileDialog::getOpenFileName(this, "Open vifo File", "J:/science data/4 Combustion/jet_0051/",
		tr("VIFO (*.vifo )"));
	if (fileName.isEmpty())
		return;
	infoFileName = fileName.toStdString();
	readInfoFile(infoFileName, data_number, datatype, dimension, space, file_list);
	file_path = file_list[0].substr(0, file_list[0].find_last_of('.'));

	loadVolume();
}


void VolumeToLabel::loadVolume()
{
	SourceVolume source_volume(file_list, dimension.x, dimension.y, dimension.z, datatype);

	//source_volume.loadVolume();	//origin data
	source_volume.loadRegularVolume(); //[0, 255] data

	volume_data = *source_volume.getRegularVolume(0);

	std::cout << "The regular volume data has been loaded." << std::endl;

	parameter_control_widget->ui.spinBox_XDim->setValue(dimension.x);
	parameter_control_widget->ui.spinBox_XDim->setEnabled(false);
	parameter_control_widget->ui.spinBox_YDim->setValue(dimension.y);
	parameter_control_widget->ui.spinBox_YDim->setEnabled(false);
	parameter_control_widget->ui.spinBox_ZDim->setValue(dimension.z);
	parameter_control_widget->ui.spinBox_ZDim->setEnabled(false);

	parameter_control_widget->ui.spinBox_Slice_index->setMaximum(dimension.z - 1);


	slice_view->updateImage(volume_data, dimension, plane_mode, slice_id);

	is_drawed = true;
}