/* Function - Displays train data, count of occurances of each product requirement in requirements and count of occurances of its label
        Pass In: train data & test data (.csv)
        Pass Out: interactive graphs
    Endfunction */

import Papa from 'papaparse';
import { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function Home() {
  const [chartData, setChartData] = useState({
    datasets: []
  });
  const [chartOptions, setChartOptions] = useState({});
  const [labelCountData, setLabelCountData] = useState([]);
  const [clickedLabelIndex, setClickedLabelIndex] = useState(null);
  const [showDoubleBarGraph, setShowDoubleBarGraph] = useState(true);

  useEffect(() => {
    Papa.parse('https://roibackend.shaktilab.org/static/data.csv', {
      download: true,
      header: true,
      dynamicTyping: true,
      delimiter: "",
      complete: (result) => {
        const labelCounts = {};

        result.data.forEach((item) => {
          const label = item['req1Product'];

          if (label) {
            if (labelCounts[label]) {
              labelCounts[label] += 1;
            } else {
              labelCounts[label] = 1;
            }
          }
        });

        setChartData({
          labels: Object.keys(labelCounts),
          datasets: [
            {
              label: "Count",
              data: Object.values(labelCounts),
              borderColor: "white",
              backgroundColor: "#ff4040"
            }
          ]
        });

        setChartOptions({
          responsive: true,
          plugins: {
            legend: {
              position: 'top'
            },
            title: {
              display: true,
              text: "Count of req1Product Values",
              color: 'black'
            }
          },
          scales: {
            x: {
              ticks: {
                color: 'black'
              }
            },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Count",
                color: 'black'
              },
              ticks: {
                color: 'black'
              }
            }
          }
        });
      }
    });
  }, []);

  const handleColumnClick = async (event, activeElements) => {
    if (activeElements.length > 0) {
      const labelIndex = activeElements[0].index;
      setClickedLabelIndex(labelIndex);

      const clickedLabel = chartData.labels[labelIndex];

      if (showDoubleBarGraph) {

        const predictionLabelCounts = {};
        await new Promise((resolve, reject) => {
          Papa.parse('https://roibackend.shaktilab.org/static/data.csv', {
            download: true,
            header: true,
            dynamicTyping: true,
            delimiter: "",
            complete: (result) => {
              result.data.forEach((item) => {
                const prediction = item['Label'];
                const label = item['req1Product'];

                if (label === clickedLabel && prediction !== undefined) {
                  if (predictionLabelCounts[prediction]) {
                    predictionLabelCounts[prediction] += 1;
                  } else {
                    predictionLabelCounts[prediction] = 1;
                  }
                }
              });

              const labelCountChartData = {
                labels: Object.keys(predictionLabelCounts),
                datasets: [
                  {
                    label: "Count",
                    data: Object.values(predictionLabelCounts),
                    borderColor: "white",
                    backgroundColor: ["#63B665", "#0476D0","#C8D9F0"]
                  }
                ]
              };

              const labelCountChartOptions = {
                responsive: true,
                plugins: {
                  legend: {
                    position: 'top'
                  },
                  title: {
                    display: true,
                    text: `Label Count for ${clickedLabel}`,
                    color: 'black'
                  }
                },
                scales: {
                  x: {
                    ticks: {
                      color: 'black'
                    }
                  },
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: "Count",
                      color: 'black'
                    },
                    ticks: {
                      color: 'black'
                    }
                  }
                }
              };

              setLabelCountData((prevData) => {
                const newData = [...prevData];
                newData[labelIndex] = {
                  chartData: labelCountChartData,
                  chartOptions: labelCountChartOptions
                };
                return newData;
              });

              resolve();
            },
            error: (error) => {
              reject(error);
            }
          });
        });
      } else {

        const req2ProductCounts = {};
        await new Promise((resolve, reject) => {
          Papa.parse('https://roibackend.shaktilab.org/static/data.csv', {
            download: true,
            header: true,
            dynamicTyping: true,
            delimiter: "",
            complete: (result) => {
              result.data.forEach((item) => {
                const req2Product = item['req2Product'];
                const label = item['req1Product'];

                if (label === clickedLabel && req2Product) {
                  if (req2ProductCounts[req2Product]) {
                    req2ProductCounts[req2Product] += 1;
                  } else {
                    req2ProductCounts[req2Product] = 1;
                  }
                }
              });

              const req2ProductChartData = {
                labels: Object.keys(req2ProductCounts),
                datasets: [
                  {
                    label: "Count",
                    data: Object.values(req2ProductCounts),
                    borderColor: "white",
                    backgroundColor: ["#ff4040", "#B6E4EB","#ffc500"]
                  }
                ]
              };

              const req2ProductChartOptions = {
                responsive: true,
                plugins: {
                  legend: {
                    position: 'top'
                  },
                  title: {
                    display: true,
                    text: `Req2Product Count for ${clickedLabel}`,
                    color: 'black'
                  }
                },
                scales: {
                  x: {
                    ticks: {
                      color: 'black'
                    }
                  },
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: "Count",
                      color: 'black'
                    },
                    ticks: {
                      color: 'black'
                    }
                  }
                }
              };

              setLabelCountData((prevData) => {
                const newData = [...prevData];
                newData[labelIndex] = {
                  chartData: req2ProductChartData,
                  chartOptions: req2ProductChartOptions
                };
                return newData;
              });

              resolve();
            },
            error: (error) => {
              reject(error);
            }
          });
        });
      }
    }
  };

  const handleCountButtonClick = () => {
    setShowDoubleBarGraph(!showDoubleBarGraph);
  };

  return (
      <>
        {chartData ? (
            <>
              <div className='canvas-cover graph-subsection'>
                <br></br>
                <Bar options={{ onClick: handleColumnClick, ...chartOptions }}  data={chartData} />
              </div>
              {labelCountData.length > 0 && labelCountData[clickedLabelIndex] && (
                  <div className='canvas-cover graph-subsection'>
                    {/* <button onClick={handleCountButtonClick} className='button--primary--blue'>
                      {showDoubleBarGraph ? "Interdependency among other products" : "Individual Feature Count"}
                    </button> */}
                    <div className='toggle--button'>
                      <button onClick={handleCountButtonClick} className={`toggle--btn--option ${showDoubleBarGraph ? "toggle--highlight" : ""}`}>
                        Interdependency among other products
                      </button>
                      <button onClick={handleCountButtonClick} className={`toggle--btn--option ${!showDoubleBarGraph ? "toggle--highlight" : ""}`}>
                        Individual Feature Count
                      </button>
                    </div>
                    <Bar
                        options={labelCountData[clickedLabelIndex].chartOptions}
                        data={labelCountData[clickedLabelIndex].chartData}
                    />
                  </div>
              )}
            </>
        ) : (
            <div>Loading...</div>
        )}
      </>
  );
}

export default Home;
    
