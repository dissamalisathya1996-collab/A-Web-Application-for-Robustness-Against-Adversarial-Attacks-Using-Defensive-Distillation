document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("attackForm");
    const fileInput = document.getElementById("imageUpload");
    const attackType = document.getElementById("attackType");
    const loadingMessage = document.getElementById("loadingMessage");
    const resultSection = document.getElementById("resultSection");
    
    form.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent form from refreshing the page
        
        if (!fileInput.files.length) {
            alert("Please upload an image!");
            return;
        }
        
        if (attackType.value === "") {
            alert("Please select an attack type!");
            return;
        }
        
        loadingMessage.style.display = "block"; // Show loading animation
        resultSection.innerHTML = ""; // Clear previous results
        
        let formData = new FormData();
        formData.append("image", fileInput.files[0]);
        formData.append("attack", attackType.value);
        
        fetch("/apply-attack", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingMessage.style.display = "none";
            resultSection.innerHTML = `
                <h3 style="color: #03346e;">Processed Image</h3>
                <img src="${data.attackedImage}" alt="Attacked Image" style="border: 2px solid #6eacda;">
                <p style="color: #e2e2b6; background-color: #021526; padding: 10px; border-radius: 5px;">Attack Type: ${attackType.value}</p>
            `;
        })
        .catch(error => {
            console.error("Error:", error);
            loadingMessage.style.display = "none";
            alert("Failed to process image. Please try again.");
        });
    });

    // Interactive animations
    document.querySelectorAll(".button").forEach(button => {
        button.addEventListener("mouseover", () => {
            button.style.transform = "scale(1.05)";
            button.style.backgroundColor = "#03346e";
            button.style.color = "#e2e2b6";
        });
        button.addEventListener("mouseleave", () => {
            button.style.transform = "scale(1)";
            button.style.backgroundColor = "#6eacda";
            button.style.color = "#021526";
        });
    });
});
