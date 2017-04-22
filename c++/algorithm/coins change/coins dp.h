#pragma once
#include<vector>

using namespace std;

class coins {
private:
	vector<int> denoms{};
	int         target;

public:
	coins() : target(0) {
	}

	coins(vector<int> &v){
		this->denoms = v;
	}

	coins(const coins& c) {
		this->denoms = c.denoms;
	}

	~coins() {
	}

	coins& operator=(const coins& c) {
		this->denoms = c.denoms;
	}

	void add(int den) {
		denoms.push_back(den);
	}

	void removeLast() {
		denoms.pop_back();
	}

	void setTarget(int t) {
		this->target = t;
	}

	int makeCoins(vector<int> v, int target);

	int countCoins(vector<int> v, int target);

	int getResult(const vector<int>& result);
};