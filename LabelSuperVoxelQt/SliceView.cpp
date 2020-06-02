#include "SliceView.h"
#include <iostream>
#include <QGraphicsPixmapItem>
#include "../SLIC3DSuperVoxel/SourceVolume.h"
#include <QMouseEvent>
#include <QDebug>

SliceView::SliceView(QWidget* parent)
{
	pixmap_item = nullptr;

	//setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	//setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setWindowState(windowState() ^ Qt::WindowMaximized);
	//fitInView(ite)

	setRenderHint(QPainter::Antialiasing);
	graphics_scene = new QGraphicsScene();
	setScene(graphics_scene);


	auto redPen = QPen(Qt::red, 2, Qt::SolidLine, Qt::FlatCap);
	auto yellowPen = QPen(Qt::yellow, 2, Qt::SolidLine, Qt::FlatCap);
	auto greenPen = QPen(Qt::green, 2, Qt::SolidLine, Qt::FlatCap);
	auto bluePen = QPen(Qt::blue, 2, Qt::SolidLine, Qt::FlatCap);
	auto writePen = QPen(Qt::white, 2, Qt::SolidLine, Qt::FlatCap);

	pen_array.append(redPen);
	pen_array.append(yellowPen);
	pen_array.append(greenPen);
	pen_array.append(bluePen);
	pen_array.append(writePen);


	std::cout << "Scene width and height is : \t" << graphics_scene->width() << "\t" << graphics_scene->height() << std::endl;
	std::cout << "Slice view width and height is : \t" << width() << "\t" << height() << std::endl;
}

SliceView::~SliceView()
{
}

void SliceView::mouseMoveEvent(QMouseEvent* event)
{
	if (event->buttons() & Qt::LeftButton)
	{
		if (pixmap_item)
		{

			const auto cur_point = pixmap_item->mapFromScene(event->pos()).toPoint();

			const auto last_path_position = cur_paint_path.currentPosition();

			cur_paint_path.lineTo(event->pos());

			cur_path->setPath(cur_paint_path);

			is_mouse_left_pressed = true;
		}

	}
	QGraphicsView::mouseMoveEvent(event);
}

void SliceView::mousePressEvent(QMouseEvent* event)
{
	if (paths_list.empty())
	{
		qDebug() << "Error: please create label first.";
		return;
	}

	if (event->buttons() & Qt::LeftButton)
	{
		if (pixmap_item)
		{
			qDebug() << "Global position : " << event->pos();
			qDebug() << "Local position : " << pixmap_item->mapFromScene(event->pos());

			start_point = pixmap_item->mapFromScene(event->pos()).toPoint();

			cur_path = new QGraphicsPathItem();
			cur_paint_path = QPainterPath();
			cur_path->setPen(pen_array[current_label_id % pen_array.size()]);
			cur_paint_path.moveTo(event->pos());

			cur_path->setPath(cur_paint_path);

			graphics_scene->addItem(cur_path);
		}

	}
	QGraphicsView::mousePressEvent(event);
}

void SliceView::mouseReleaseEvent(QMouseEvent* event)
{

	if (is_mouse_left_pressed)
	{
		//qDebug() << "test";
		if (pixmap_item)
		{
			paths_list[current_label_id][plane_id][slice_number].push_back(cur_path);

			qDebug() << "cur_path size : " << cur_paint_path;
			is_mouse_left_pressed = false;

			emit signal_updateLableNumber(paths_list);

		}
	}
	//qDebug() << "test2";
	QGraphicsView::mouseReleaseEvent(event);
}

void SliceView::keyPressEvent(QKeyEvent* event)
{
	switch (event->key())
	{
	case Qt::Key_Up:
	case Qt::Key_Right:
		if (pixmap_item)
		{
			signal_updateSliceId(1);
		}
		break;
	case Qt::Key_Down:
	case Qt::Key_Left:
		if (pixmap_item)
		{
			signal_updateSliceId(-1);
		}
		break;
	case Qt::Key_Control:
		is_control_key_pressed = true;
	}

	//QGraphicsView::keyPressEvent(event);
}

void SliceView::keyReleaseEvent(QKeyEvent* event)
{
	//QGraphicsView::keyReleaseEvent(event);
	switch (event->key())
	{
	case Qt::Key_Control:
		is_control_key_pressed = false;
	}

}

void SliceView::wheelEvent(QWheelEvent* event)
{

	if (pixmap_item)
	{
		if (is_control_key_pressed)	//放大缩小
		{
			//scale_point = event->posF();

			if (scale_factor <= 0) scale_factor = 0.1;
			if (event->delta() > 0)
				scale_factor += 0.1;
			else
				scale_factor -= 0.1;

			auto items = graphics_scene->items();
			for (auto item : items)
			{
				item->setTransformOriginPoint(scale_point);
				item->setScale(scale_factor);
			}

			//pixmap_item->setTransformOriginPoint(scale_point);
			//pixmap_item->setScale(scale_factor);
		}
		else
		{
			if (event->delta() > 0)
			{
				signal_updateSliceId(1);
			}
			else
			{
				signal_updateSliceId(-1);
			}
		}

	}
	//QGraphicsView::keyReleaseEvent(event);
}

void SliceView::createNewPathItemArray(const QString& label_name)
{
	QVector<QVector<QVector<QGraphicsPathItem*>>> buf;
	buf.resize(3);
	for (auto i = 0; i < 3; i++)
	{
		if (i == 0)
			buf[i].resize(dimension.z);
		else if (i == 1)
			buf[i].resize(dimension.x);
		else if (i == 2)
			buf[i].resize(dimension.y);
	}

	paths_list.push_back(buf);
	path_id_map[label_name] = paths_list.size() - 1;
}

void SliceView::setLabel(const QString& label_name)
{
	if (path_id_map.find(label_name) != path_id_map.end())
	{
		current_label_id = path_id_map[label_name];
	}
	else
	{
		qDebug() << "Error in set label.";
	}
}

void SliceView::updateImage(std::vector<unsigned char>& volume_data, const hxy::my_int3& dimension, const int& plane_id,
	const int& slice_number)
{
	this->slice_number = slice_number;
	this->plane_id = plane_id;
	this->dimension = dimension;

	for (auto i : graphics_scene->items())
	{
		graphics_scene->removeItem(i);
	}

	delete pixmap_item;

	auto pixmap = getPixImage(volume_data, dimension, plane_id, slice_number);

	setSceneRect(0, 0, width(), height());
	graphics_scene->setSceneRect(0, 0, width(), height());
	graphics_scene->setBackgroundBrush(QBrush(Qt::white));
	//centerOn(width() / 2, height() / 2);
	std::cout << "Width and height : " << width() << "\t" << height() << std::endl;
	pixmap_item = graphics_scene->addPixmap(pixmap);
	pixmap_item->setPos((width() - pixmap.width()) / 2, (height() - pixmap.height()) / 2);

	//centerOn((width() - pixmap.width()) / 2, (height() - pixmap.height()) / 2);

	//scale_point = QPointF((width() - pixmap.width()) / 2, (height() - pixmap.height()) / 2);
	scale_point = QPointF((width()) / 2.0, (height()) / 2.0);

	//pixmap_item->setTransformOriginPoint(scale_point);
	pixmap_item->setScale(scale_factor);


	//全部remove掉
	// for(auto i =0;i<paths_list.size();i++)
	// {
	// 	for(auto j=0;j<paths_list[i].size();j++)
	// 	{
	// 		for(auto k=0;k<paths_list[i][j].size();k++)
	// 		{
	// 			for(auto m=0;m<paths_list[i][j][k].size();m++)
	// 			{
	// 				graphics_scene->removeItem(paths_list[i][j][k][m]);
	// 			}
	// 		}
	// 	}
	// }



	//仅显示对应切片的数据
	for (auto i = 0; i < paths_list.size(); i++)
	{
		for (auto& j : paths_list[i][plane_id][slice_number])
		{
			j->setTransformOriginPoint(scale_point);
			j->setScale(scale_factor);
			j->setPen(pen_array[i % pen_array.size()]);
			graphics_scene->addItem(j);
		}
	}

	graphics_scene->update();
}


QPixmap SliceView::getPixImage(std::vector<unsigned char>& volume_data, const hxy::my_int3& dimension, const int& plane_mode, const int& slice_id)
{
	if (plane_mode == 0)
	{
		QImage image(dimension.x, dimension.y, QImage::Format_Grayscale8);
		auto offset = slice_id * dimension.x * dimension.y;
		for (auto i = 0; i < dimension.y; i++)
		{
			for (auto j = 0; j < dimension.x; j++)
			{
				auto buf = volume_data[offset + i * dimension.x + j];
				image.setPixel(j, i, qRgb(buf, buf, buf));
			}
		}
		return QPixmap::fromImage(image);
	}
	else if (plane_mode == 1)	//yz plane
	{
		QImage image(dimension.y, dimension.z, QImage::Format_Grayscale8);
		auto offset = slice_id;

		for (auto i = 0; i < dimension.z; i++)
		{
			for (auto j = 0; j < dimension.y; j++)
			{
				auto buf = volume_data[i * dimension.x * dimension.y + j * dimension.x + offset];
				image.setPixel(j, i, qRgb(buf, buf, buf));
			}
		}
		return QPixmap::fromImage(image);
	}
	else //xz plane
	{
		QImage image(dimension.x, dimension.z, QImage::Format_Grayscale8);
		auto offset = slice_id * dimension.x;

		for (auto i = 0; i < dimension.z; i++)
		{
			for (auto j = 0; j < dimension.x; j++)
			{
				auto buf = volume_data[i * dimension.x * dimension.y + offset + j];
				image.setPixel(j, i, qRgb(buf, buf, buf));
			}
		}
		return QPixmap::fromImage(image);
	}
}

int SliceView::getLabelId(const QString& label_name)
{
	if (path_id_map.find(label_name) != path_id_map.end())
	{
		return path_id_map[label_name];
	}
	else
	{
		return -1;
	}
}

QVector<QVector<int>>& SliceView::tranPathListToVolumeIndex()
{
	volume_index.resize(paths_list.size());
	auto yz_rect = QRect(0, 0, dimension.y, dimension.z);
	auto xz_rect = QRect(0, 0, dimension.x, dimension.z);
	auto xy_rect = QRect(0, 0, dimension.x, dimension.y);

	for (auto i = 0; i < volume_index.size(); i++)
	{
		//xy plane
		{

			auto& path_list = paths_list[i][0];
			for (auto j = 0; j < path_list.size(); j++) //对应切片
			{
				//j为对应的切片
				for (auto& k : path_list[j]) //对应切片上的每条路径
				{
					auto path = k->path();
					auto local_path = pixmap_item->mapFromScene(path);
					for (auto id = 0; id < local_path.elementCount(); id++)
					{
						auto point = QPointF(local_path.elementAt(id)).toPoint();
						if (xy_rect.contains(point))
						{
							//计算原始点
							volume_index[i].push_back(dimension.x * dimension.y * j + point.y() * dimension.x + point.x());
						}
					}
				}
			}
		}
		//yz plane
		{
			auto& path_list = paths_list[i][1];
			for (auto j = 0; j < path_list.size(); j++) //对应切片
			{
				//j为对应的切片
				for (auto& k : path_list[j]) //对应切片上的每条路径
				{
					auto path = k->path();
					auto local_path = pixmap_item->mapFromScene(path);
					for (auto id = 0; id < local_path.elementCount(); id++)
					{
						auto point = QPointF(local_path.elementAt(id)).toPoint();
						if (yz_rect.contains(point))
						{
							//计算原始点
							volume_index[i].push_back(dimension.x * dimension.y * point.y() + point.x() * dimension.x + j);
						}
					}
				}
			}
		}
		//xz plane
		{
			auto& path_list = paths_list[i][2];
			for (auto j = 0; j < path_list.size(); j++) //对应切片
			{
				//j为对应的切片
				for (auto& k : path_list[j]) //对应切片上的每条路径
				{
					auto path = k->path();
					auto local_path = pixmap_item->mapFromScene(path);
					for (auto id = 0; id < local_path.elementCount(); id++)
					{
						auto point = QPointF(local_path.elementAt(id)).toPoint();
						if (xz_rect.contains(point))
						{
							//计算原始点
							volume_index[i].push_back(dimension.x * dimension.y * point.y() + j * dimension.x + point.x());
						}
					}
				}
			}
		}
	}

	return volume_index;
}
