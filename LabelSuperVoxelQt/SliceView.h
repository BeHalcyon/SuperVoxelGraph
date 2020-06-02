#pragma once
#include <QGraphicsView>
#include "../SLIC3DSuperVoxel/SourceVolume.h"
#include <QMap>


class SliceView :
	public QGraphicsView
{
	Q_OBJECT
public:
	SliceView(QWidget* parent = nullptr);
	~SliceView();

	void mouseMoveEvent(QMouseEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;

	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;

	void wheelEvent(QWheelEvent* event) override;

	void createNewPathItemArray(const QString& label_name);
	void setLabel(const QString& label_name);

	void updateImage(std::vector<unsigned char>& volume_data, const hxy::my_int3& dimension, const int& plane_id, const int& slice_number);
	QPixmap getPixImage(std::vector<unsigned char>& volume_data, const hxy::my_int3& dimension, const int& plane_mode,
		const int& slice_id);

	int getLabelId(const QString& label_name);

	QVector<QVector<int>>& tranPathListToVolumeIndex();
signals:
	void signal_updateLableNumber(const QVector<QVector<QVector<QVector<QGraphicsPathItem*>>>>& vector);
	void signal_updateSliceId(int offset);
private:
	QGraphicsScene* graphics_scene;
	QGraphicsPixmapItem* pixmap_item;
	bool						is_control_key_pressed = false;
	double						scale_factor = 1.0;
	QPointF						scale_point;

	QVector<QVector<QVector < QVector<QGraphicsPathItem*>>>>		paths_list;
	QMap<QString, int>			path_id_map;
	int							current_label_id = 0;

	QGraphicsPathItem* cur_path;
	int							slice_number;
	int							plane_id;
	hxy::my_int3				dimension;
	QPoint						start_point;
	QPainterPath				cur_paint_path;
	QVector<QPen>				pen_array;
	bool						is_mouse_left_pressed = false;

	QVector<QVector<int>>		volume_index;
};

