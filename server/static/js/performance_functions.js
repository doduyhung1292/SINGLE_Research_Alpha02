// Function to fetch symbol performance data
function fetchSymbolPerformanceData() {
    console.log("Fetching symbol performance data...");
    fetch('/api/symbol_performance')
        .then(response => {
            console.log("symbol performance API response status:", response.status);
            if (!response.ok) {
                throw new Error(`symbol performance API error (${response.status}): ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("symbol performance data received:", data);
            if (data && data.symbol_performance && data.symbol_performance.length > 0) {
                updateSymbolPerformanceChart(data.symbol_performance);
                updatePerformanceTables(data.symbol_performance);
            } else {
                console.warn("No symbol performance data available or empty data received");
                document.getElementById('symbol-performance-chart-container').innerHTML =
                    '<div class="alert alert-warning">No symbol performance data available</div>';
                document.getElementById('symbol-performance-table').innerHTML =
                    '<tr><td colspan="6" class="text-center">No data available</td></tr>';
            }
        })
        .catch(error => {
            console.error("Error loading symbol performance data:", error);
            document.getElementById('symbol-performance-chart-container').innerHTML =
                `<div class="alert alert-danger">Error loading symbol performance data: ${error.message}</div>`;
            document.getElementById('symbol-performance-table').innerHTML =
                `<tr><td colspan="6" class="text-danger">Error: ${error.message}</td></tr>`;
        });
}

// Function to update the symbol performance chart
function updateSymbolPerformanceChart(symbolData) {
    const canvas = document.getElementById('symbol-performance-chart');
    
    if (!canvas) {
        console.error("Cannot find canvas element #symbol-performance-chart");
        return;
    }
    
    // Get the 2d context from the canvas
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
        console.error("Cannot get 2d context from canvas");
        return;
    }
    
    // Clear previous chart if exists
    if (window.symbolPerformanceChart) {
        window.symbolPerformanceChart.destroy();
    }

    // Check if we have data
    if (!symbolData || symbolData.length === 0) {
        document.getElementById('symbol-performance-chart-container').innerHTML = 
            '<div class="alert alert-warning">No performance data available</div>';
        return;
    }

    // Prepare datasets
    const datasets = [];
    const colors = [
        'rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)',
        'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)'
    ];

    symbolData.forEach((symbol, index) => {
        if (symbol.data && symbol.data.length > 0) {
            const color = colors[index % colors.length];
            datasets.push({
                label: symbol.name || `Symbol ${index + 1}`,
                data: symbol.data.map(point => ({
                    x: point[0], // Trade number
                    y: point[1]  // Cumulative PnL
                })),
                borderColor: color,
                backgroundColor: color.replace('1)', '0.2)'),
                borderWidth: 2,
                tension: 0.1
            });
        }
    });

    // Create new chart only if we have datasets
    if (datasets.length > 0) {
        try {
            window.symbolPerformanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Trade Number'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Cumulative P&L'
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            mode: 'nearest',
                            intersect: false
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        title: {
                            display: true,
                            text: 'Alpha Trading by Symbols Cumulative Performance'
                        }
                    }
                }
            });
        } catch (error) {
            console.error("Error creating chart:", error);
            document.getElementById('symbol-performance-chart-container').innerHTML = 
                `<div class="alert alert-danger">Failed to create chart: ${error.message}</div>`;
        }
    } else {
        document.getElementById('symbol-performance-chart-container').innerHTML = 
            '<div class="alert alert-warning">No chart data available</div>';
    }
}

// Function to update performance tables
function updatePerformanceTables(symbolPerformanceData) {
    console.log("Updating performance tables with data:", symbolPerformanceData);

    // Update symbol stats table
    const symbolStatsBody = document.getElementById('symbol-performance-table');
    if (symbolStatsBody) {
        symbolStatsBody.innerHTML = '';

        if (!symbolPerformanceData || symbolPerformanceData.length === 0) {
            symbolStatsBody.innerHTML = '<tr><td colspan="6" class="text-center">No data available</td></tr>';
            // Clear overall stats too if no data
            document.getElementById('total-all-symbols-return').textContent = '0.00';
            document.getElementById('avg-all-symbols-return').textContent = '0.00';
            document.getElementById('avg-all-symbols-sharpe').textContent = '0.00';
            document.getElementById('overall-win-rate').textContent = '0.00%';
            return;
        }

        // Process and sort the data by total return (descending)
        const processedData = [];
        let totalOverallReturn = 0;
        let totalTrades = 0;
        let totalWinningTrades = 0;

        symbolPerformanceData.forEach(symbol => {
            if (symbol.name && symbol.data && symbol.data.length > 0) {
                console.log(`Processing symbol ${symbol.name} with ${symbol.data.length} data points`);

                const symbolTrades = symbol.data.length;
                // Correctly calculate winning trades based on PnL of each trade, not cumulative
                // Assuming point[1] is cumulative PnL, we need individual trade PnL if available
                // For now, let's approximate based on positive cumulative steps, which isn't ideal
                // A better approach needs individual trade results from the backend
                let symbolWinningTrades = 0;
                let lastPnl = 0;
                symbol.data.forEach(point => {
                    if (point[1] > lastPnl) {
                        symbolWinningTrades++;
                    }
                    lastPnl = point[1];
                });

                const winRate = symbolTrades > 0 ? (symbolWinningTrades / symbolTrades) * 100 : 0;
                const finalReturn = symbol.data[symbol.data.length - 1][1];
                const avgReturn = symbolTrades > 0 ? finalReturn / symbolTrades : 0;

                processedData.push({
                    name: symbol.name,
                    trades: symbolTrades,
                    winRate: winRate,
                    avgReturn: avgReturn,
                    totalReturn: finalReturn,
                    sharpeRatio: 'N/A' // Sharpe calculation needs risk-free rate and return stddev
                });

                totalOverallReturn += finalReturn;
                totalTrades += symbolTrades;
                totalWinningTrades += symbolWinningTrades; // Summing approximate wins

            } else {
                console.warn("Skipping symbol with invalid data:", symbol);
            }
        });

        // Sort by total return (descending)
        processedData.sort((a, b) => b.totalReturn - a.totalReturn);

        console.log(`Processed ${processedData.length} symbols for display`);

        // Render the sorted data
        processedData.forEach(item => {
            const row = document.createElement('tr');
            const totalReturnClass = item.totalReturn >= 0 ? 'text-success' : 'text-danger';

            // Add row class for negative return
            if (item.totalReturn < 0) {
                row.className = 'table-danger';
            }

            row.innerHTML = `
                <td>${item.name}</td>
                <td>${item.trades}</td>
                <td>${item.winRate.toFixed(2)}%</td>
                <td>${item.avgReturn.toFixed(6)}</td>
                <td class="${totalReturnClass}">${item.totalReturn.toFixed(6)}</td>
                <td>${item.sharpeRatio}</td>
            `;
            symbolStatsBody.appendChild(row);
        });

        // Calculate and update overall stats
        const avgOverallReturn = processedData.length > 0 ? totalOverallReturn / processedData.length : 0;
        const overallWinRate = totalTrades > 0 ? (totalWinningTrades / totalTrades) * 100 : 0;
        // Sharpe Ratio calculation is complex and needs more data (risk-free rate, std dev of returns)
        const avgOverallSharpe = 'N/A';

        document.getElementById('total-all-symbols-return').textContent = totalOverallReturn.toFixed(2);
        document.getElementById('avg-all-symbols-return').textContent = avgOverallReturn.toFixed(2);
        document.getElementById('avg-all-symbols-sharpe').textContent = avgOverallSharpe; // Update if calculated
        document.getElementById('overall-win-rate').textContent = `${overallWinRate.toFixed(2)}%`;

    } else {
        console.error("Could not find performance table element");
    }
}

// Add this logic after the config modal form is shown for editing or creating a symbol
$(document).ready(function() {
    // When opening the config modal for editing, populate tradeSymbol
    $(document).on('click', '.edit-config', function() {
        const configId = $(this).data('id');
        $.get(`/api/config/${configId}`, function(config) {
            $('#config-id').val(config._id);
            $('#symbol').val(config.symbol);
            // ... populate other fields ...
        });
    });

    // When saving the config, include tradeSymbol in the payload
    $('#save-config').on('click', function() {
        const configId = $('#config-id').val();
        const payload = {
            symbolA: $('#symbol').val(),
            // ... collect other fields ...
        };
        // ... existing code for AJAX POST/PUT ...
    });
});