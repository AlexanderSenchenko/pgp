set terminal png size 800, 650 enhanced font 'Times New Roman, 16'
set title ""

set key top left

set pointsize 1

# set ylabel "Время, сек"  font 'Times New Roman, 16'
# set xlabel "Размер вектора"  font 'Times New Roman, 16'

# set xrange [ 0 : 8388608 ]

# set xtics ("1<<10" 1024,\
# 			"1<<11" 2048,\
# 			"1<<12" 4096,\
# 			"1<<13" 8192,\
# 			"1<<14" 16384,\
# 			"1<<15" 32768,\
# 			"1<<16" 65536,\
# 			"1<<17" 131072,\
# 			"1<<18" 262144,\
# 			"1<<19" 524288,\
# 			"1<<20" 1048576,\
# 			"1<<21" 2097152,\
# 			"1<<22" 4194304,\
# 			"1<<23" 8388608)

set grid xtics lc rgb  '#555555' lw 1 lt 0
set grid ytics lc rgb  '#555555' lw 1 lt 0

set key left top

set style line 1 linetype 1 linewidth 1 linecolor rgb 'red'
set style line 2 linetype 2 linewidth 1 linecolor rgb 'blue'
set style line 3 linetype 3 linewidth 1 linecolor rgb 'green'
set style line 4 linetype 4 linewidth 1 linecolor rgb 'purple'
set style line 5 linetype 5 linewidth 1 linecolor rgb 'cyan'
set style line 6 linetype 6 linewidth 1 linecolor rgb 'magenta'
set style line 7 linetype 7 linewidth 1 linecolor rgb 'orange'
set style line 8 linetype 8 linewidth 1 linecolor rgb 'black'

set output "data_sin.png"
# plot "1024_1.txt" u 1:2 w linespoints ls 1 pt 7 title "blocks: 1024 threads: 1",\
	# "512_2.txt" u 1:2 w linespoints ls 2 pt 5 title "blocks: 512 threads: 2",\
	# "128_8.txt" u 1:2 w linespoints ls 3 pt 4 title "blocks: 128 threads: 128",\
	# "32_32.txt" u 1:2 w linespoints ls 4 pt 8 title "blocks: 32 threads: 32",\
	# "8_128.txt" u 1:2 w linespoints ls 5 pt 6 title "blocks: 8 threads: 128"

plot "data_sin" u 1:2 w linespoints ls 1 pt 7 title "sin"
