public class Main {
public static void main(String[] args) {
final int capacity = 100;
int[] arr = new int[capacity];
arr[0] = 4;
arr[1] = 5;
arr[2] = 6;
arr[3] = 7;
arr[4] = 8;
arr[5] = 9;
arr[6] = 10;
int n = 7;
int k = 3;
buildHeap(arr, n, k);
System.out.println("Built Heap: ");
for (int i = 0; i < n; i++)
System.out.print(arr[i] + " ");
int element = 3;
insert(arr, n, k, element);
n++;
System.out.println("Heap after insertion of " +element + ": ");
for (int i = 0; i < n; i++)
System.out.print(arr[i] + " ");
System.out.println("Extracted max is " +extractMax(arr,n,k));
n--;
System.out.println("\n\nHeap after extract max: ");
for (int i = 0; i < n; i++)
System.out.print(arr[i] + " ");
}
public static void buildHeap(int[] arr, int n, int k) {
for (int i = (n - 1) / k; i >= 0; i--)
restoreDown(arr, n, i, k);
}
public static void insert(int[] arr, int n, int k, int elem) {
arr[n - 1] = elem;
restoreUp(arr, n - 1, k);
}
public static int extractMax(int[] arr, int n, int k) {
int max = arr[0];arr[0] = arr[n - 1];
restoreDown(arr, n - 1, 0, k);
return max;
}

public static void restoreDown(int[] arr, int len, int index, int k){
int[] child = new int[k + 1];
while (true) {
for (int i = 1; i <= k; i++)
child[i]=(k*index+i) < len ? (k*index+i) : -1;
int maxChild = -1, maxChildIndex = 0;
for (int i = 1; i <= k; i++) {
if (child[i] != -1 && arr[child[i]] > maxChild) {
maxChildIndex = child[i];
maxChild = arr[child[i]];
}
}
if (maxChild == -1)
break;
if (arr[index] < arr[maxChildIndex])
swap(arr, index, maxChildIndex);
index = maxChildIndex;
}
}

public static void restoreUp(int[] arr, int index, int k) {
int parent = (index - 1) / k;
while (parent >= 0) {
if (arr[index] > arr[parent]) {
swap(arr, index, parent);
index = parent;
parent = (index - 1) / k;
} else
break;
}
}
public static void swap(int[] arr, int i, int j) {
int temp = arr[i];
arr[i] = arr[j];
arr[j] = temp;
}
}