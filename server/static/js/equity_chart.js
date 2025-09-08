// Register zoom plugin
try {
    // For Chart.js v3+
    if (typeof ChartZoom !== 'undefined') {
        Chart.register(ChartZoom);
        console.log('ChartZoom plugin registered');
    } else {
        console.warn('ChartZoom plugin not found. Zoom functionality will not be available.');
    }
} catch (e) {
    console.error('Error registering ChartZoom plugin:', e);
}

// Khởi tạo biểu đồ equity
function initEquityChart() {
    console.log('Initializing equity chart with Chart.js');
    // Tạo biểu đồ trống trước
    const ctx = document.getElementById('equity-chart-container');

    // Xóa nội dung cũ nếu đã tồn tại
    if (ctx.innerHTML !== '') {
        ctx.innerHTML = '';
    }

    // Tạo canvas cho Chart.js
    const canvas = document.createElement('canvas');
    canvas.id = 'equity-chart';
    ctx.appendChild(canvas);

    // Khởi tạo biểu đồ trống với nhiều datasets
    window.equityChart = new Chart(canvas, {
        type: 'line',
        data: {
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Equity Components'
                    },
                    beginAtZero: false
                },
                y1: {
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Total Equity'
                    },
                    beginAtZero: false,
                    grid: {
                        drawOnChartArea: false // chỉ vẽ grid cho trục y bên trái
                    }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy'
                    },
                    zoom: {
                        wheel: {
                            enabled: true
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy'
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });

    // Add reset zoom button
    const resetZoomBtn = document.createElement('button');
    resetZoomBtn.className = 'btn btn-sm btn-outline-secondary mt-2';
    resetZoomBtn.innerHTML = 'Reset Zoom';
    resetZoomBtn.onclick = function() {
        window.equityChart.resetZoom();
    };
    ctx.parentNode.insertBefore(resetZoomBtn, ctx.nextSibling);

    // Fetch dữ liệu
    fetchEquityData();
}

// Lấy dữ liệu biểu đồ equity
function fetchEquityData() {
    console.log('Fetching equity data...');
    $.ajax({
        url: '/api/equity_chart',
        method: 'GET',
        success: function (response) {
            console.log('Equity data response:', response);
            if (response && response.equity_data) {
                updateEquityChart(response.equity_data);
                
                // Update correlation display if data exists
                if (response.correlation) {
                    updateCorrelationDisplay(response.correlation);
                }
            } else {
                console.error('Invalid equity data response format:', response);
            }
        },
        error: function (xhr, status, error) {
            console.error('Error fetching equity data:', error);
            console.error('Response:', xhr.responseText);
        }
    });
}

// Cập nhật biểu đồ equity sử dụng Chart.js
function updateEquityChart(equityData) {
    console.log('Updating equity chart with data', equityData);

    if (!window.equityChart) {
        console.error('Chart not initialized');
        initEquityChart();
        return;
    }

    // Kiểm tra dữ liệu
    if (!equityData || !Array.isArray(equityData)) {
        console.error('Invalid equity data format');
        return;
    }

    // Xóa dữ liệu cũ
    window.equityChart.data.datasets = [];

    // Màu mặc định cho các datasets
    const colors = [
        'rgb(75, 192, 192)',    // Teal for Alpha01
        'rgb(255, 99, 132)',    // Red for Alpha02
        'rgb(54, 162, 235)'     // Blue for Total Equity
    ];
    const bgColors = [
        'rgba(75, 192, 192, 0.1)',
        'rgba(255, 99, 132, 0.1)',
        'rgba(54, 162, 235, 0.1)'
    ];

    // Thêm từng dataset vào biểu đồ
    equityData.forEach((dataset, index) => {
        if (!dataset || !dataset.data || !Array.isArray(dataset.data)) {
            console.error(`Invalid dataset format at index ${index}`, dataset);
            return;
        }

        // Chuyển đổi dữ liệu từ [[timestamp, value], ...] sang định dạng của Chart.js
        const chartData = dataset.data.map(point => {
            // Đảm bảo timestamp được xử lý đúng
            let timestamp = point[0];
            // Nếu timestamp không phải là số, chuyển đổi nó
            if (typeof timestamp === 'string') {
                timestamp = new Date(timestamp).getTime();
            }
            return {
                x: new Date(timestamp),
                y: point[1]
            };
        });

        // Đặt màu và label rõ ràng cho từng đường
        let color = colors[index % colors.length];
        let bgColor = bgColors[index % bgColors.length];
        let label = dataset.name || `Dataset ${index + 1}`;
        
        // Cấu hình đặc biệt cho Total Equity - làm đậm đường
        let borderWidth = 2;
        let yAxisID = 'y'; // Mặc định sử dụng trục y bên trái
        
        if (label === 'Total Equity') {
            borderWidth = 4; // Làm đậm đường Total Equity
            yAxisID = 'y1'; // Sử dụng trục y bên phải cho Total Equity
        }
        
        window.equityChart.data.datasets.push({
            label: label,
            data: chartData,
            borderColor: color,
            backgroundColor: bgColor,
            tension: 0.1,
            borderWidth: borderWidth,
            pointRadius: 1,
            pointHoverRadius: 5,
            yAxisID: yAxisID // Chỉ định trục y cho dataset
        });
    });

    // Cập nhật cấu hình biểu đồ
    window.equityChart.options.plugins.title = {
        display: true,
        text: 'Portfolio Equity'
    };

    // Cập nhật biểu đồ
    window.equityChart.update();
}
// Function to update the correlation display
function updateCorrelationDisplay(correlation) {
    const correlationValue = correlation.value;
    const windowSize = correlation.window;
    const sampleSize = correlation.sample_size;
    
    // Update the UI with correlation information
    const correlationElement = document.getElementById('btc-equity-correlation');
    const windowElement = document.getElementById('correlation-window');
    const sampleElement = document.getElementById('correlation-sample-size');
    
    if (correlationValue !== null) {
        // Format the correlation value
        correlationElement.textContent = correlationValue.toFixed(4);
        
        // Add color based on correlation strength
        correlationElement.classList.remove('text-success', 'text-danger', 'text-warning', 'text-info');
        
        if (correlationValue > 0.7) {
            correlationElement.classList.add('text-success');
        } else if (correlationValue < -0.7) {
            correlationElement.classList.add('text-danger');
        } else if (Math.abs(correlationValue) > 0.3) {
            correlationElement.classList.add('text-warning');
        } else {
            correlationElement.classList.add('text-info');
        }
        
        // Update window information
        windowElement.textContent = `(${windowSize} period rolling window)`;
        
        // Update sample size
        sampleElement.textContent = `Based on ${sampleSize} data points`;
    } else {
        correlationElement.textContent = 'Not enough data';
        correlationElement.classList.remove('text-success', 'text-danger', 'text-warning', 'text-info');
        correlationElement.classList.add('text-muted');
        windowElement.textContent = '';
        sampleElement.textContent = 'Insufficient data points available';
    }
}

$(document).ready(function() {
    // Khởi tạo chart mặc định là live
    initEquityChart();

    // Sự kiện chuyển đổi giữa live/simulation
    $('#btn-live-equity').on('click', function() {
        $(this).addClass('active btn-outline-primary').removeClass('btn-outline-secondary');
        $('#btn-sim-equity').removeClass('active btn-outline-primary').addClass('btn-outline-secondary');
        fetchEquityData();
    });
    $('#btn-sim-equity').on('click', function() {
        $(this).addClass('active btn-outline-primary').removeClass('btn-outline-secondary');
        $('#btn-live-equity').removeClass('active btn-outline-primary').addClass('btn-outline-secondary');
        fetchSimulationEquityData();
    });
});

function fetchSimulationEquityData() {
    console.log('Fetching simulation equity data...');
    $.ajax({
        url: '/api/simulation_equity_chart',
        method: 'GET',
        success: function (response) {
            console.log('Simulation equity data response:', response);
            if (response && response.equity_data) {
                updateEquityChart(response.equity_data);
            } else {
                console.error('Invalid simulation equity data response format:', response);
            }
        },
        error: function (xhr, status, error) {
            console.error('Error fetching simulation equity data:', error);
            console.error('Response:', xhr.responseText);
        }
    });
}