#include <algorithm>
#include <iostream>
#include "coins dp.h"

using namespace std;

int min(int a, int b) {
	return (a < b ? a : b);
}

int coins::makeCoins(vector<int> v, int target) {
	if (target == 0) {
		return 0;
	}

	vector<int> result{ 999999 };

	for (auto val : v) {
		if (val <= target) {
			result.push_back(countCoins(v, target));
		}
	}

	int count = getResult(result);

	return count;
}

int coins::countCoins(vector<int> v, int target) {
	if (v.size() == 0) {
		return 999999;
	}

	int c = v.back();
	v.pop_back();
	vector<int> result{ 999999 };
	result.push_back(countCoins(v, target - c));
	int count = getResult(result);
	return count;
}

int coins::getResult(const vector<int>& result) {
	auto count = min_element(result.begin(), result.end());

	int num = *count;

	cout << "min num: " << num << endl;

	return num;
}

int main() {
	vector<int> denoms{1, 6, 10};

	const int target = 12;

	coins c(denoms);

	auto count = c.makeCoins(denoms, target);

	cout << "Final: " << count << endl;

	int waiting;

	cin >> waiting;
}