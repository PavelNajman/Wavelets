#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

size_t pso(long pos, size_t size)
{
	if( pos < 0 )
		pos *= -1;
	if( pos > (long)size-1 )
		pos = 2*((long)size-1) - pos;

	return (size_t)pos;
}

void transform_line(float *begin, size_t stride, size_t size)
{
	float alpha = -1.0f;
	float beta  = +.5f;

	for(long x = 0; x < (long)size; x += 2)
	{
		size_t r = pso(x+1, size);

		begin[stride*((size_t)x)] -= beta  * begin[stride*r];
	}

	for(long x = 1; x < (long)size; x += 2)
	{
		size_t l = pso(x-1, size);

		begin[stride*((size_t)x)] -= alpha * begin[stride*l];
	}
}

void dump(float *data, size_t size_x, size_t size_y, const char *path)
{
	FILE *file = fopen(path, "w");

	assert(file);

	for(size_t y = 0; y < size_y; y++)
	{
		for(size_t x = 0; x < size_x; x++)
		{
			fprintf(file, "%+f%s", data[y*size_x+x], (x == size_x - 1) ? "\n" : ",\t");
		}
	}

	fclose(file);
}

void load(float *data, size_t size_x, size_t size_y, const char *path)
{
	FILE *file = fopen(path, "r");

	assert(file);

	for(size_t y = 0; y < size_y; y++)
	{
		for(size_t x = 0; x < size_x; x++)
		{
			assert( 1 == fscanf(file, "%f, ", data + y*size_x + x) );
		}
	}

	fclose(file);
}

int main()
{
	size_t size_x = 512, size_y = 512;

	float *data = malloc(size_x * size_y * sizeof(float));

	assert(data);

	load(data, size_x, size_y, "output.mat");
	dump(data, size_x, size_y, "loaded.mat");

	// transform rows
	for(size_t y = 0; y < size_y; y++)
	{
		transform_line(data + y*size_x, 1, size_x);
	}

	// transform cols
	for(size_t x = 0; x < size_x; x++)
	{
		transform_line(data + x*1, size_x, size_y);
	}

	dump(data, size_x, size_y, "inverse.mat");

	free(data);

	return 0;
}
