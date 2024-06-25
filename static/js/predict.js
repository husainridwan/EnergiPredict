// Global variables
let currentPage = 1;
const rowsPerPage = 10;
let predictionsData = [];

// Function to display the selected file name
function displayFileName() {
    const fileInput = document.getElementById('fileInput');
    const fileNameContainer = document.getElementById('fileNameDisplay'); // Changed 'fileName' to 'fileNameDisplay'
    if (fileInput.files.length > 0) {
        fileNameContainer.textContent = "Selected file: " + fileInput.files[0].name;
    } else {
        fileNameContainer.textContent = "No file selected"; // Changed from empty string to a more descriptive message
    }
}

// Function to upload the file
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict', { // Changed endpoint from '/predict-fault' to '/api/predict'
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Update predictions data
        predictionsData = data.predictions; // Assuming the response structure changed to include predictions within a predictions key
        // Render predictions table
        renderPredictionsTable();
        // Enable download button
        document.getElementById('predictionsContainer').style.display = 'block';
        // Show pagination
        document.querySelector('.pagination-controls').style.display = 'block'; // Changed class from '.pagination' to '.pagination-controls'
        // Enable download button
        document.getElementById('downloadBtn').disabled = false; // Changed from changing display style to disabling/enabling the button
    })
    .catch(error => {
        // Display error message alert
        console.error('Upload failed:', error); // Changed from alert to console.error for better debugging
    });
}

// Function to render predictions table
function renderPredictionsTable() {
    const predictionsTable = document.getElementById('predictionsTableBody'); // Changed 'predictionsTable' to 'predictionsTableBody'
    predictionsTable.innerHTML = ''; // Clear previous content

    // Calculate start and end indices for current page
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;

    // Assuming the structure of predictionsData or the way it's displayed has changed, adjust accordingly
    predictionsData.slice(startIndex, endIndex).forEach(prediction => {
        const row = predictionsTable.insertRow();
        Object.values(prediction).forEach(text => {
            const cell = row.insertCell();
            cell.textContent = text;
        });
    });
}

// Function to go to previous page
function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        renderPredictionsTable();
    }
}

// Function to go to next page
function nextPage() {
    if (currentPage < Math.ceil(predictionsData.length / rowsPerPage)) {
        currentPage++;
        renderPredictionsTable();
    }
}

// Function to download predictions as CSV
function downloadPredictions() {
    // Add header row
    let csvContent = Object.keys(predictionsData[0]).join(",") + "\n";

    // Add data rows
    predictionsData.forEach(row => {
        csvContent += Object.values(row).join(",") + "\n";
    });

    // Create a Blob with the CSV content
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });

    // Create a link element
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'predictions.csv'; // Set download filename

    // Simulate a click on the link to trigger download
    link.click();

    // Revoke the temporary URL object after download
    URL.revokeObjectURL(link.href);
}

document.getElementById('downloadBtn').addEventListener('click', downloadPredictions);

// Get the file input element and the upload button element
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');

// Add an event listener to the file input change event
fileInput.addEventListener('change', displayFileName);

// Add an event listener to the upload button click event
uploadBtn.addEventListener('click', uploadFile);

document.addEventListener('DOMContentLoaded', function () {
    // Assuming there are changes to the features or descriptions, update accordingly
    const requiredFeatures = [
        "mels_S", "lig_S", "mels_N", "air_temp_set_1", "relative_humidity_set_1", 
        "solar_radiation_set_1", "intTemp", "extTemp", "airSpeed", "waterHeat", 
        "hp_hws_temp", "rtuSat", "rtuRat", "rtuMat", "rtuOat", "rtuFan", "hvac_total"
    ];

    const featureDescriptions = [
        // Assuming descriptions have changed or need to be updated
        "Miscellaneous electric load for the South Wing of the building",
        "Lighting load for the South Wing",
        "Miscellaneous electric load for the North Wing of the building",
        "Outdoor air temperature from sensor 1",
        "Relative humidity from sensor 1",
        "Solar radiation from sensor 1",
        "Zone temperature of interior zone",
        "Zone temperature of exterior zone",
        "Supply air fan speed of Zones",
        "Heating water valve position of Zones",
        "Heat pump heating water supply temperature",
        "Roof Top Unit supply air temperature setpoint",
        "Roof Top Unit return air temperature",
        "Roof Top Unit mixed air temperature",
        "Roof Top Unit outdoor air temperature",
        "Roof Top Unit supply fan speed",
        "Total HVAC load for the building"
    ];

    const tableBody = document.querySelector("#featuresTable tbody");

    requiredFeatures.forEach((feature, index) => {
        const description = featureDescriptions[index];
        const row = document.createElement("tr");
        const featureCell = document.createElement("td");
        featureCell.textContent = feature;
        const descriptionCell = document.createElement("td");
        descriptionCell.textContent = description;
        row.appendChild(featureCell);
        row.appendChild(descriptionCell);
        tableBody.appendChild(row);
    });
});