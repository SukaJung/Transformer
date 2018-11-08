package com.example.suka.transformer;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.net.http.AndroidHttpClient;
import android.os.AsyncTask;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;


import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.HttpMultipartMode;
import org.apache.http.entity.mime.MultipartEntity;
import org.apache.http.entity.mime.content.ByteArrayBody;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;

public class CameraActivity extends AppCompatActivity {
    Button btn;
    private HttpClient client;
    ImageView iv;
    final int REQUEST_LOAD_IMAGE = 1,MY_CAMERA_REQUEST_CODE = 100;
    ImageRequest imageRequest;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        iv = (ImageView) findViewById(R.id.iv);
        setup();
    }
    private void setup()
    {
        btn = (Button)findViewById(R.id.camera_btn);


        btn.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                Log.e("click","");
                request_Permission_Camera();

            }
        });
    }
    public void request_Permission_Camera(){
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
        {
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA))
            {

            }
            else
            {
                ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE );

                // MY_PERMISSIONS_REQUEST_READ_CONTACTS is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        }
        else{
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(intent,1);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        Bitmap img= null;
        if (requestCode == REQUEST_LOAD_IMAGE && resultCode == RESULT_OK && data != null){
            if(data.getData()==null){
                img = (Bitmap)data.getExtras().get("data");
            }
            else{
                try{
                    img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
                }catch(IOException e){
                    Log.e("IOException",e.toString());
                }
            }
//            Uri selectedImage = data.getData();
//            iv.setImageURI(data.getData());
            iv.setImageBitmap(img);
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

//            Cursor cursor = getContentResolver().query(img,
//                    filePathColumn, null, null, null);
//            cursor.moveToFirst();
//
//            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
//            String picturePath = cursor.getString(columnIndex);
//            cursor.close();
//            img = BitmapFactory.decodeFile(picturePath);
//            int width = img.getWidth();
//            int height = img.getHeight();
//            Matrix matrix = new Matrix();
//            matrix.postRotate(270);
//            Bitmap resizedBitmap = Bitmap.createBitmap(img, 0, 0, width, height, matrix, true);
//            img.recycle();

            imageRequest = new ImageRequest();
            imageRequest.execute(img);
        }
    }

    private class ImageRequest extends AsyncTask<Bitmap, Void, Bitmap> {
        ProgressDialog asyncDialog = new ProgressDialog(
                CameraActivity.this);

        public ImageRequest(){
            client = AndroidHttpClient.newInstance("Android");
        }

        @Override
        protected void onPreExecute(){
            asyncDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
            asyncDialog.setMessage("전송중입니다..");

            // show dialog
            asyncDialog.show();
            super.onPreExecute();
        }
        @Override
        protected Bitmap doInBackground(Bitmap... bitmaps) {
            Bitmap bitmap = bitmaps[0];
//            saveCameraImage(bitmap);
            MultipartEntity reqEntity = new MultipartEntity(HttpMultipartMode.BROWSER_COMPATIBLE);

            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 75, bos);
            byte[] data = bos.toByteArray();
            ByteArrayBody bab = new ByteArrayBody(data, "image.png");

            Log.e("A","B");
            HttpPost post = new HttpPost("http://218.235.176.143:8000/upload");
            try{
                reqEntity.addPart("file", bab);

                post.setEntity(reqEntity);
                HttpResponse httpRes;
                httpRes = client.execute(post);
                HttpEntity resEntity = httpRes.getEntity();

                if (resEntity != null) {
                    Log.e("STATE", "resEntity != null");
                    InputStream is = resEntity.getContent();
                    bitmap = BitmapFactory.decodeStream(is);
                    return bitmap;
                }
                else {
                    Log.e("result", "is null");
                }
                if (resEntity != null) {
                    resEntity.consumeContent();
                }
            } catch (Exception e){
                e.printStackTrace();
            }

            return null;
        }

        private String readAll(Reader rd) throws IOException {
            StringBuilder sb = new StringBuilder();
            int cp;
            while ((cp = rd.read()) != -1) {
                sb.append((char) cp);
            }
            return sb.toString();
        }

        @Override
        protected void onPostExecute(Bitmap bitmap){
            super.onPostExecute(bitmap);

            Log.e("STATE", "onPostExecute");
            iv.setImageBitmap(bitmap);
//            saveTempImage(bitmap);
            client.getConnectionManager().shutdown();
            asyncDialog.dismiss();
            Intent intent = new Intent(CameraActivity.this, ResultActivity.class);
            startActivity(intent);

        }
    }

}
