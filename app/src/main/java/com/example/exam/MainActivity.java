package com.example.exam;

import androidx.annotation.Nullable;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.exam.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    Button camara;
    Button NextBtn;
    int imageSize = 224;
    public static final int sub = 1001;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        camara = findViewById(R.id.button2);

        camara.setOnClickListener(new View.OnClickListener() { //카메라 클릭 리스터
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });




        picture.setOnClickListener(new View.OnClickListener() { //갤러리 클릭 리스터
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image){  //분류함수
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel=0;

            for(int i=0; i<imageSize; i++){
                for(int j=0; j<imageSize; j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val>>16) & 0xFF) * (1.f/255.f));
                    byteBuffer.putFloat(((val>>8) & 0xFF) * (1.f/255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f/255.f));

                }
            }



            inputFeature0.loadBuffer(byteBuffer);  //바이트버퍼에 저장

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer(); //추론 진행하기

            float[] confidences= outputFeature0.getFloatArray(); //각자의 신뢰도 값 배열에 저장
            int maxPos=0;
            float maxConfidence=0;
            for(int i=0; i<confidences.length; i++){
                if(confidences[i]>maxConfidence){
                    maxConfidence= confidences[i];
                    maxPos=i;
                } //가장 높은 신뢰도 찾기
            }
            String[] classes={"우치","랑이","삼색이","우동이"};
            result.setText(classes[maxPos]); //가장 높은 신뢰도값 결과창에 출력

            String s="";
            for(int i=0; i<classes.length; i++){
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] *100);
                confidence.setText(s); //각각의 신뢰도값 결과창에 출력
            }
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
        //분석 후 카메라나 갤러리 버튼 사라지기
        camara.setVisibility(View.INVISIBLE);
        picture.setVisibility(View.INVISIBLE);
        //분석 후 손동작 분석 사이트로 이동 버튼 생성
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3 ){ //카메라 버튼 눌렀을때 이미지 처리 코드 3을 받음
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);  //치수에 맞게 이미지 크기 조정
                imageView.setImageBitmap(image); //화면에 선택이미지 띄우기기

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false); //이미지를 비트맵으로 가져오기
                classifyImage(image); //분류함수에 이미지 보내기

            }


            else{ //갤러리 버튼 눌렀을때 이미지 처리 (코드 1을 받음)
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);


    }


}