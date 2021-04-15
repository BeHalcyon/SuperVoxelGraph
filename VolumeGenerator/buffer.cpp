#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
using namespace std;

vector<string> smalldata;

vector<string> quchong(vector<string>& list1) {
    vector<string> list2;
    for (int i = 0; i < list1.size(); i++) {
        bool flag = false;
        for (int j = 0; j < list2.size(); j++) {
            if (list1[i] == list2[j]) {
                flag = true;
            }
        }
        if (flag == false) {
            list2.push_back(list1[i]);
        }
    }
    return list2;
}

void getdata(vector<vector<string>>& bigdata) {

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                for (int m = 0; m < 10; m++) {
                    vector<string> data;
                    if (i == 0 && j == 0) data.push_back("NN");
                    else if (k == 0 && m == 0) data.push_back("NN");
                    else if (i == 0 && j != 0) {
                        if (k == 0 && m == 0) {
                            data.push_back(to_string(i) + to_string((j + m) % 10) + to_string(k) + to_string(m));
                        }
                        else if (m == 0 && k != 0) {
                            data.push_back(to_string(i) + to_string((j + k) % 10) + to_string(k) + to_string(m));
                        }
                        else {
                            data.push_back(to_string(i) + to_string((j + k) % 10) + to_string(k) + to_string(m));
                            data.push_back(to_string(i) + to_string((j + m) % 10) + to_string(k) + to_string(m));
                        }
                    }
                    else if (i != 0 && j == 0) {
                        if (k == 0 && m != 0) {
                            data.push_back(to_string((i + m) % 10) + to_string(j) + to_string(k) + to_string(m));
                        }
                        else if (m == 0 && k != 0) {
                            data.push_back(to_string((i + k) % 10) + to_string(j) + to_string(k) + to_string(m));
                        }
                        else {
                            data.push_back(to_string((i + k) % 10) + to_string(j) + to_string(k) + to_string(m));
                            data.push_back(to_string((i + m) % 10) + to_string(j) + to_string(k) + to_string(m));
                        }
                    }
                    else if (i != 0 && j != 0) {
                        if (k == 0 && m != 0) {
                            data.push_back(to_string((i + m) % 10) + to_string(j) + to_string(k) + to_string(m));
                            data.push_back(to_string(i) + to_string((j + m) % 10) + to_string(k) + to_string(m));
                        }
                        else if (m == 0 && k != 0) {
                            data.push_back(to_string((i + k) % 10) + to_string(j) + to_string(k) + to_string(m));
                            data.push_back(to_string(i) + to_string((j + k) % 10) + to_string(k) + to_string(m));
                        }
                        else {
                            data.push_back(to_string((i + k) % 10) + to_string(j) + to_string(k) + to_string(m));
                            data.push_back(to_string((i + m) % 10) + to_string(j) + to_string(k) + to_string(m));
                            data.push_back(to_string(i) + to_string((j + k) % 10) + to_string(k) + to_string(m));
                            data.push_back(to_string(i) + to_string((j + m) % 10) + to_string(k) + to_string(m));
                        }
                    }
                    data = quchong(data);
                    bigdata.push_back(data);
                }
            }
        }
    }
}


void fuc(vector<vector<string>>& bigdata, int n, string num)
{
	if (num != "NN")
	{
		for (int t=0;t<bigdata[stoi(num)].size();t++)
		{
            int c = n;
            c -= 1;
            if (c > 0) {
                string buffer = bigdata[stoi(num)][t];
                string invert(buffer.rbegin(), buffer.rend());
                fuc(bigdata, c, invert);
            }
            else
            {
	            for (int s=0;s<bigdata[stoi(num)].size();s++)
	            {
                    string buffer = bigdata[stoi(num)][s];
                    string invert(buffer.rbegin(), buffer.rend());
                    smalldata.push_back(invert);
	            }
                //cout << smalldata.size()<< " "<<  stoi(num) << endl;
                return;
            }
		}
        n -= 1;
	}
}

int count(vector<string>& f)
{
    int co = 0;
	for (int i=0;i<f.size();i++)
	{
        if (f[i][2] == '0' && f[i][3] == '0') co++;
	}
    return co;
}


double winornot(vector<vector<string>>& bigdata, vector<string>& d, int y)
{
    int win = 1, lose = 1;
	for (int q=1;q<11;q++)
	{
        fuc(bigdata, q, d[y]);
        vector<string> f = smalldata;
        smalldata.clear();
        int co = count(f);
        if (q % 2 == 1)
            lose += co;
        else
            win += co;
	}
    return win*1.0 / lose;
}

int cl(vector<float>& listw, float nc)
{
	for(int i=0;i<listw.size();i++)
	{
        if (abs(listw[i] - nc) < 1e-5) return i;
	}
    return -1;
}
int cl(vector<double>& listw)
{
    double max_value = -1;
    int max_index = 0;
    for (int i = 0; i < listw.size(); i++)
    {
	    if (max_value < listw[i])
	    {
            max_value = listw[i];
            max_index = i;
	    }
    }
    return max_index;
}


void run(vector<vector<string>>& bigdata)
{
    string x;
    cout << "Please input nums:";
    cin >> x;
    string s(x.rbegin(), x.rend());
    vector<double> winstuff;
	for (int y=0;y<bigdata[stoi(s)].size();y++)
	{
        fuc(bigdata, 1, s);
        vector<string> d = smalldata;
        smalldata.clear();
        double g = winornot(bigdata, d, y);
        winstuff.push_back(g);
	}
    fuc(bigdata, 1, s);
    vector<string> d = smalldata;
    smalldata.clear();
    cout << d[cl(winstuff)] << endl;
	
}

int _main() {
    while (true) {
        vector<vector<string>> bigdata;
        getdata(bigdata);
        smalldata.clear();
        run(bigdata);
    }
    return 0;
}