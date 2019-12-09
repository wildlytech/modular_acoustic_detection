(function() {
    registerPrintButtonHandler();
    return;

    function registerPrintButtonHandler() {
        var button = document.getElementById("download-link-report");

        if (!button || button.onclick === onPrintButtonClick) {
            setTimeout(registerPrintButtonHandler, 200);
            return;
        }

        button.onclick = onPrintButtonClick;
    }

    function onPrintButtonClick() {
        window.print();
    }
})();
