using Godot;
using System;
using System.Text;

public class main : PanelContainer
{
	bool isHttpReq;
	static HTTPRequest httpRequest;
	string httpURL;
	string[] httpHeaders;
	string httpBody;
	Label connectLabel;
	string subjectSelect;
	float learningRate;

	T GetValueArray<T>(Godot.Collections.Array ary, params int[] coor)
	{
		Godot.Collections.Array arySub = ary;
		for (int i=0;i<coor.Length-1;i++) {
			arySub = (Godot.Collections.Array) arySub[coor[i]];
		}
		return (T) arySub[coor[coor.Length-1]];
	}

	ImageTexture ConvertMat2Texture(Godot.Collections.Array size, Godot.Collections.Array mat)
	{
		Image img = new Image();
		int width = (int) ((float) size[0]);
		int height = (int) ((float) size[1]);
		img.Create(width, height, false, (Image.Format) 5);
		img.Lock();
		Godot.Collections.Array arySubX;
		Godot.Collections.Array arySubY;
		for (int x=0;x<width;x++){
			arySubX = (Godot.Collections.Array) mat[x];
			for (int y=0;y<height;y++){
				arySubY = (Godot.Collections.Array) arySubX[y];
				Color c = img.GetPixel(y,x);
				c.r = ((float) arySubY[2])/255f;
				c.b = ((float) arySubY[0])/255f;
				c.g = ((float) arySubY[1])/255f;
				c.a = 1.0f;
				img.SetPixel(y, x, c);
			}
		}
		img.Unlock();
		ImageTexture groundTexture = new ImageTexture();
		groundTexture.CreateFromImage(img);
		return groundTexture;
	}

	Image ConvertAlpha(Image img, Godot.Collections.Array mat)
	{
		img.Lock();
		int width = img.GetWidth();
		int height = img.GetHeight();
		for (int x=0;x<width;x++){
			for (int y=0;y<height;y++){
				Color c = img.GetPixel(x,y);
				float alpha = GetValueArray<float>(mat, y, x);
				if (alpha > 0.7f) {
					c.a = 1.0f;
				} else if (alpha > 0.4f) {
					c.a = 0.5f;
				} else {
					c.a = 0.2f;
				}
				img.SetPixel(x,y,c);
			}
		}
		img.Unlock();
		return img;
	}

	async void RequestInitConnect(Godot.Collections.Dictionary reqDic)
	{
		httpURL = (string) reqDic["httpURL"];
		connectLabel.Text = "Connected";
	}

	async void RequestApplyChangeSubject(Godot.Collections.Dictionary reqDic)
	{
		Godot.Collections.Array size = (Godot.Collections.Array) reqDic["size"];
		Godot.Collections.Array matTrain = (Godot.Collections.Array) reqDic["matTrain"];
		Godot.Collections.Array matTest = (Godot.Collections.Array) reqDic["matTest"];

		TextureRect textureRectTrain = GetNode<TextureRect>("MainHBox/ImageHBox/ImageTrainVBox/TextureTrain");
		textureRectTrain.SetTexture(ConvertMat2Texture(size, matTrain));
		TextureRect textureRectTest = GetNode<TextureRect>("MainHBox/ImageHBox/ImageTestVBox/TextureTest");
		textureRectTest.SetTexture(ConvertMat2Texture(size, matTest));
	}

	async void RequestApplyTrainModel(Godot.Collections.Dictionary reqDic)
	{
		Godot.Collections.Array size = (Godot.Collections.Array) reqDic["size"];
		Godot.Collections.Array matTrain = (Godot.Collections.Array) reqDic["matTrain"];
		Godot.Collections.Array mskTest = (Godot.Collections.Array) reqDic["mskTest"];

		TextureRect textureRectTrain = GetNode<TextureRect>("MainHBox/ImageHBox/ImageTrainVBox/TextureTrain");
		textureRectTrain.SetTexture(ConvertMat2Texture(size, matTrain));
		
		TextureRect textureRectTest = GetNode<TextureRect>("MainHBox/ImageHBox/ImageTestVBox/TextureTest");
		Texture text = textureRectTest.GetTexture();
		Image imgOld = text.GetData();
		Image imgNew = ConvertAlpha(imgOld, mskTest);
		ImageTexture groundTexture = new ImageTexture();
		groundTexture.CreateFromImage(imgNew);
		textureRectTest.SetTexture(groundTexture);
	}

	void RequestApplyResetModel(Godot.Collections.Dictionary reqDic)
	{
		
	}
	
	async void RequestApplyTestChange(Godot.Collections.Dictionary reqDic)
	{
		Godot.Collections.Array size = (Godot.Collections.Array) reqDic["size"];
		Godot.Collections.Array matTrain = (Godot.Collections.Array) reqDic["matTrain"];
		Godot.Collections.Array matTest = (Godot.Collections.Array) reqDic["matTest"];
		Godot.Collections.Array mskTest = (Godot.Collections.Array) reqDic["mskTest"];

		TextureRect textureRectTrain = GetNode<TextureRect>("MainHBox/ImageHBox/ImageTrainVBox/TextureTrain");
		textureRectTrain.SetTexture(ConvertMat2Texture(size, matTrain));
		TextureRect textureRectTest = GetNode<TextureRect>("MainHBox/ImageHBox/ImageTestVBox/TextureTest");
		textureRectTest.SetTexture(ConvertMat2Texture(size, matTest));

		Texture text = textureRectTest.GetTexture();
		Image imgOld = text.GetData();
		Image imgNew = ConvertAlpha(imgOld, mskTest);
		ImageTexture groundTexture = new ImageTexture();
		groundTexture.CreateFromImage(imgNew);
		textureRectTest.SetTexture(groundTexture);
	}

	void OnHTTPRequestRequestCompleted(int result, int response_code, string[] headers, byte[] body)
	{
		if (response_code == 200) {
			var json = JSON.Parse(Encoding.UTF8.GetString(body));
			var json2 = JSON.Parse((string) json.Result);
			Godot.Collections.Dictionary reqDic = (Godot.Collections.Dictionary) json2.Result;
			string req = (string) reqDic["req"];

			if (req == "Init") {
				RequestInitConnect(reqDic);
			} else if (req == "SubjectChange") {
				RequestApplyChangeSubject(reqDic);
			} else if (req == "TrainModel") {
				RequestApplyTrainModel(reqDic);
			} else if (req == "TestChange") {
				RequestApplyTestChange(reqDic);
			} else if (req == "ResetModel") {
				RequestApplyResetModel(reqDic);
			}
		} else {
			connectLabel.Text = "Error";
		}

		isHttpReq = false;
	}

	void SendHTTPInitConnect(string ipAddress)
	{
		string httpURLTemp = "http://" + ipAddress;
		httpBody = JSON.Print(new Godot.Collections.Dictionary
		{
			{ "req", "Init" },
			{ "httpURL", httpURLTemp }
		});
		httpRequest.Request(httpURLTemp+"/init", httpHeaders, true, HTTPClient.Method.Post, httpBody);
	}

	void SendHTTPSubjectChange(string subject)
	{
		subjectSelect = subject;
		httpBody = JSON.Print(new Godot.Collections.Dictionary
		{
			{ "req", "SubjectChange" },
			{ "subject", subjectSelect }
		});
		httpRequest.Request(httpURL+"/subject/change", httpHeaders, true, HTTPClient.Method.Post, httpBody);
	}

	void SendHTTPTrainModel()
	{
		httpRequest.Request(httpURL+"/train/next", httpHeaders, true, HTTPClient.Method.Get);
	}

	void SendHTTPTestChange()
	{
		httpRequest.Request(httpURL+"/test/change", httpHeaders, true, HTTPClient.Method.Get);
	}

	void SendHTTPLRUpdate()
	{
		httpBody = JSON.Print(new Godot.Collections.Dictionary
		{
			{ "req", "ResetModel" },
			{ "learningRate", learningRate }
		});
		httpRequest.Request(httpURL+"/change/lr", httpHeaders, true, HTTPClient.Method.Post, httpBody);
	}

	void SendHTTPResetModel()
	{
		httpBody = JSON.Print(new Godot.Collections.Dictionary
		{
			{ "req", "ResetModel" },
			{ "learningRate", learningRate }
		});
		httpRequest.Request(httpURL+"/reset/model", httpHeaders, true, HTTPClient.Method.Post, httpBody);
	}

	private void OnIPLineEditTextEntered(String ipAddress)
	{
		if (isHttpReq == false) {
			isHttpReq = true;
			SendHTTPInitConnect(ipAddress);
		}
	}

	void OnSubjectButtonPressed(string subject)
	{
		if (isHttpReq == false) {
			isHttpReq = true;
			SendHTTPSubjectChange(subject);
		}
	}

	void OnTrainModelButtonPressed()
	{
		if (isHttpReq == false) {
			isHttpReq = true;
			SendHTTPTrainModel();
		}
	}

	void OnTestChageButtonPressed()
	{
		if (isHttpReq == false) {
			isHttpReq = true;
			SendHTTPTestChange();
		}
	}

	void OnLRHScrollBarValueChanged(float lr)
	{
		learningRate = (float) Math.Pow(10, lr);
		if (isHttpReq == false) {
			isHttpReq = true;
			SendHTTPLRUpdate();
		}
	}

	void OnResetModelButtonPressed()
	{
		if (isHttpReq == false) {
			isHttpReq = true;
			SendHTTPResetModel();
		}
	}

	void InitHTTPRequest()
	{
		isHttpReq = false;
		httpRequest = GetNode<HTTPRequest>("HTTPRequest");
		httpHeaders = new string[] {"Access-Control-Allow-Origin: *"};
		connectLabel = GetNode<Label>("MainHBox/VBoxContainer/ConnectLabel");
		connectLabel.Text = "Ready";
	}

	public override void _Ready()
	{
		InitHTTPRequest();
		learningRate = 0.0001f;
		OnIPLineEditTextEntered(GetNode<LineEdit>("MainHBox/VBoxContainer/IPLineEdit").Text);
	}
}
