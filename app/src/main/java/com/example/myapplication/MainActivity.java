package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private Bitmap bitmap1;
    private ArrayList<double[]> lines_final = new ArrayList<>();
    private Bitmap tmpbitmap;

    //openCV4Android 需要加载用到
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
//                    mOpenCvCameraView.enableView();
//                    mOpenCvCameraView.setOnTouchListener(ColorBlobDetectionActivity.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    //用于查看每一步生成的bitmap效果
    private void tmpBitmap(Mat src){
        tmpbitmap = Bitmap.createBitmap(src.width(), src.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(src, tmpbitmap);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //动态加载opencv库
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        bitmap1 = BitmapFactory.decodeResource(MainActivity.this.getResources(),R.mipmap.test);
        Log.d(TAG, "成功生成bitmap");
        Bitmap bitmap = convert(bitmap1);
        ImageView img = findViewById(R.id.image_1);
        img.setImageBitmap(bitmap);
        Log.d(TAG,"设置成功");
    }

    public Bitmap convert(Bitmap bitmap){
        //记得把图像new Mat改为三通道rgb
        Mat mat_src = new Mat();
        Utils.bitmapToMat(bitmap, mat_src);     //将bitmap转化成mat_src
        Mat frame = new Mat();
        Core.copyMakeBorder(mat_src, frame,5,5,5,5, Core.BORDER_CONSTANT);
        Imgproc.cvtColor(frame,frame,Imgproc.COLOR_BGR2GRAY);
//        tmpBitmap(frame);
        Mat img1 = new Mat();
        Mat img2 = new Mat();
        //下面这个部分不确定两个数值参数与py是否一致，完整编写后再debug
        Imgproc.threshold(frame, img1, 50, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(frame, img2, 150, 255, Imgproc.THRESH_BINARY);
//        tmpBitmap(img2);

        //膨胀操作
        Mat dst = new Mat();
        Log.d(TAG, "img2的宽度"+img2.width());
        Imgproc.dilate(img2,dst,new Mat());      //如果要注明Iterations的话就会需要anchor，而python中没有anchor
//        tmpBitmap(dst);

        //图像的选择需要考虑
        Mat edges= new Mat();
        Imgproc.Canny(img1,edges,0,255,3);
//        tmpBitmap(edges);     //edges没有问题
        ArrayList<double[]> lines_point = CalcDegree(edges, frame);

        //用于检测linespoint是否出现问题，这里检测是2个，应该没有问题
        Log.d(TAG, "检测linespoint"+lines_point.size());
        Log.d(TAG, "检测第一个点 x0: " + lines_point.get(0)[0] + "; y0:" + lines_point.get(0)[1]+
                                           " ; x1:" +lines_point.get(0)[2]+" ;y1:"+lines_point.get(0)[3]);
        Log.d(TAG, "检测第二个点 x0: " + lines_point.get(1)[0] + "; y0:" + lines_point.get(1)[1]+
                " ; x1:" +lines_point.get(1)[2]+" ;y1:"+lines_point.get(1)[3]);


        if (lines_point.size() < 2) {
            Log.d("MainActivity", "没有检测到线条");
        }

        Point[] points1 = {new Point(lines_point.get(0)[0], lines_point.get(0)[1]),new Point(lines_point.get(0)[2],lines_point.get(0)[3]),
                new Point(lines_point.get(1)[2],lines_point.get(1)[3]),new Point(lines_point.get(1)[0], lines_point.get(1)[1]) };
        MatOfPoint2f pts_src = new MatOfPoint2f(points1);
        Log.d(TAG, "查看pts_src" + pts_src.dump());
//        pts_src.fromArray(points1);
        Point[] points2 = {new Point(0, 0), new Point(299, 0), new Point(299, 99), new Point(0,99)};
        MatOfPoint2f pts_dst = new MatOfPoint2f(points2);
        Log.d(TAG, "查看pts_dst" + pts_dst.dump());
//        pts_dst.fromArray(points2);
        Mat h = Calib3d.findHomography(pts_src, pts_dst);
        Log.d(TAG, "findHomography获得的h矩阵是："+h.dump());
//        tmpBitmap(h);     好像不支持h的图片类型
        Mat im_out = new Mat();


//        tmpBitmap(img2);
        Imgproc.warpPerspective(img2,im_out,h, new Size(300,100));   //这个部分与python端使用imgg定义方法不同
        //尝试下变img2(乱玩)
//        Imgproc.resize(img2, img2, new Size(600, img2.height()), Imgproc.INTER_LINEAR);

        Imgproc.resize(im_out, im_out, new Size(450, 100), Imgproc.INTER_LINEAR);
        tmpBitmap(im_out);
        Imgproc.erode(im_out, im_out, new Mat(), new Point(-1,-1), 1);
        Core.bitwise_not(im_out,im_out);
        Bitmap result = Bitmap.createBitmap(im_out.width(), im_out.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(im_out, result);
        return result;
    }

    private ArrayList<double[]> CalcDegree(Mat srcImage, Mat rgbimage) {
        ArrayList<double[]> linese_up = null;
        ArrayList<double[]> lines_down = null;
        int[] match_beiyong = null;
        int[] match_line = null;
        double min_distance = 10000;
        double x3 ;
        double y3;
        double x4;   //  原来里面获取img.shaep[1]就是图片的宽，
        double y4;
        double k = 0;
        Mat lineimage = new Mat();
        //霍夫变换检测直线，第五个参数表示阈值，阈值越大，精确度越高？
        Mat lines = new Mat();
        Imgproc.HoughLines(srcImage,lines,1,Math.PI/180,70);
        Log.d(TAG, "霍夫变换检测直线"+lines.toString());
        int sum=0;
        ArrayList<double[]> lines_point = new ArrayList<>();
//        if(lines==null){
//            return new double[0, 0];
//        }
        if (lines.rows() > 0 && lines.cols() > 0) {
            for (int i = 0; i < lines.cols(); i++) {
                double[] data = lines.get(0, i);
                double rho = data[0];
                double theta = data[1];
                double a = Math.cos(theta);
                double b = Math.sin(theta);
                double x0 = a*rho;
                double y0 = b*rho;
                double x1 = (int)Math.round(x0+1000*(-b));
                double y1 = (int)Math.round(x0+1000*a);
                double x2 = (int)Math.round(x0-1000*(-b));
                double y2 = (int)Math.round(x0-1000*a);

//                计算斜率    假如有很多线，咋办？ 只需要挑两条出来，两条线要满足斜率最接近，同时一条在上面一条在下面
//                排除除数为0的情况
                if(x2!=x1){
                    b = -(y2-y1)/(x2-x1)*x1+y1;
                    k = (y2-y1)/(x2-x1);
                }else{
                    continue;
                }
                x3 = 0;
                y3 = (int)x3 * k + b;
                x4 = rgbimage.width();   //  原来里面获取img.shaep[1]就是图片的宽，
                y4 = (int) (x4 * k + b);
                double[] point = {x3,y3,x4,y4};
                lines_point.add(point);
                sum+=theta;
                Imgproc.line(lineimage,new Point(x3,y3),new Point(x4,y4),new Scalar(0,0,255),1,Imgproc.LINE_AA);
            }
        }
        //过滤一些线条
        //因为java一个方法返回两种数据类型比较麻烦，这里直接将lines_final声明为成员变量
        if(lines_point.size()>=2){
//            创建两个集合，一个存放上面的线，一个存放下面的线
            linese_up = new ArrayList<>();
            lines_down = new ArrayList<>();
            //遍历所有线，进行分类
            for(int i=0;i<lines_point.size();i++){
                if(lines_point.get(i)[1] < (double) srcImage.height()/2){
                    linese_up.add(lines_point.get(i));
                }else{
                    lines_down.add(lines_point.get(i));
                }
            }
            //现在保证上下都只有一根线
            int upline = linese_up.size();
            int lowline = lines_down.size();
            //上下有一边没线，挑一根出来，斜率最大的那个
            int index_line = 0;
            if(upline == 0 || lowline == 0){
                lines_point.clear();
                if(upline>0){
                    double max_k = CalSlope(linese_up.get(0));
                    for(int i=0;i<upline;i++){
                        double k1 = CalSlope(linese_up.get(i));
                        if(max_k < k1){
                            max_k = k1;
                            index_line = i;
                        }
                    }
                    lines_point.add(linese_up.get(index_line));
                }else {
                    double max_k = CalSlope(lines_down.get(0));
                    for (int i = 0; i < upline; i++) {
                        double k1 = CalSlope(lines_down.get(i));
                        if (max_k < k1) {
                            max_k = k1;
                            index_line = i;
                        }
                    }
                    lines_point.add(lines_down.get(index_line));
                }
            }else{
                //       这样，lines_point 有一条线
//        上下都有线的情况
//        计算两条线斜率最接近，有时候不太好
//        所以考虑是不是可以增加斜率最大的?
//        或者是把几条线取平均？
                for(int i=0; i<lines_down.size();i++){
                    for(int j=0;j<linese_up.size();j++){
                        double k1 = CalSlope(linese_up.get(i));
                        double k2 = CalSlope(lines_down.get(j));
                        //斜率异号
                        if((k1*k2)<0){
                            continue;
                        }
                        double distance_k = CalSlopeDistance(k1, k2);
                        if(distance_k<min_distance){
//                        如果有一方是小于0.02，说明基本平行，滤除，实在没有采用
                            if(Math.abs(k1)<0.02||Math.abs(k2)<0.02){
                                match_beiyong = new int[]{i, j};
                            }else{
                                match_line = new int[]{i,j};
                            }
                            min_distance = distance_k;
                        }
                    }
                }
                //看滤除0.02斜率之后的线有没有匹配得到，如果有匹配到就使用，匹配不成功就降低要求
                if(match_line.length!=0){
                    lines_final.add(linese_up.get(match_line[0]));
                    lines_final.add(lines_down.get(match_line[1]));
                }else{
                    lines_final.add(linese_up.get(match_beiyong[0]));
                    lines_final.add(lines_down.get(match_beiyong[1]));
                }
            }
        }
//       对所有角度求平均，这样做旋转效果会更好
        double average = sum/lines.width();
        double angle = DegreeTrans(average) - 90;
        if(lines_point.size()==1){
            if(lines_point.get(0)[1]>srcImage.height()/2){
                x3 = 0;
                y3 = x3*k+1;
                x4 = rgbimage.width();
                y4 = x4*k+1;
                double[] point = {x3,y3,x4,y4};
                lines_point.add(point);
                Imgproc.circle(lineimage, new Point(x3, y3), 3, new Scalar(255, 0, 0), -1);
                Imgproc.circle(lineimage, new Point(x4, y4), 3, new Scalar(255, 0, 0), -1);
                Imgproc.line(lineimage,new Point(x3,y3),new Point(x4,y4),new Scalar(0,0,255),1,Imgproc.LINE_AA);
                //互换位置
                Collections.swap(lines_point,0,1);
                lines_final.clear();
                lines_final.add(lines_point.get(0));
                lines_final.add(lines_point.get(1));
            }else{
                k = CalSlope(lines_point.get(0));
                x4 = srcImage.width();
                y4 = srcImage.height();
                int b = (int) (y4 - x4 * k);
                x3 = 0;
                y3 = 0*k+b;
                double[] point = {x3,y3,x4,y4};
                lines_point.add(point);
                lines_final.clear();
                lines_final.add(lines_point.get(0));
                lines_final.add(lines_point.get(1));
            }
        }

        //画最后的线
        if (lines_final.size() == 2) {
            for(int i=0;i<2;i++){
                x3 = lines_final.get(i)[0];
                y3 = lines_final.get(i)[1];
                x4 = lines_final.get(i)[2];
                y4 = lines_final.get(i)[3];
                Imgproc.line(lineimage, new Point(x3, y3), new Point(x4, y4), new Scalar(0.255, 0), Imgproc.LINE_AA);
            }
        }
        return lines_point;

    }

    private double DegreeTrans(double theta) {
        return theta / Math.PI * 180;
    }

    private double CalSlopeDistance(double k1, double k2) {
        return Math.abs(k1 - k2);
    }

    private double CalSlope(double[] line) {
        double x1 = line[0];
        double y1 = line[1];
        double x2 = line[2];
        double y2 = line[3];
        double b = -(y2 - y1) / (x2 - x1) * x1 + y1;
        double k = (y2 - y1) / (x2 - x1);
        return k;
    }


}