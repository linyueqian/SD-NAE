var isCNSamplesVisible = false; // Tracks if samples are currently visible

function toggleSamples() {
    // Get all elements with class 'more-samples'
    var samples = document.getElementsByClassName('more-samples');
    
    // If samples are currently visible, hide them
    if (isCNSamplesVisible) {
        for (var i = 0; i < samples.length; i++) {
            samples[i].style.display = 'none';
        }
        document.getElementsByClassName('toggle-samples-button')[0].innerHTML = '<i class="fas fa-plus fa-fw"></i>More Samples';
        isCNSamplesVisible = false;
    } 
    // If samples are currently hidden, show them
    else {
        for (var i = 0; i < samples.length; i++) {
            samples[i].style.display = 'block';
        }
        document.getElementsByClassName('toggle-samples-button')[0].innerHTML = '<i class="fas fa-chevron-up fa-fw"></i>Hide Samples';
        isCNSamplesVisible = true;
    }
}

var isENSamplesVisible = false; // Tracks if samples are currently visible

function toggleENSamples() {
    // Get all elements with class 'more-ensamples'
    var samples = document.getElementsByClassName('more-ensamples');
    
    // If samples are currently visible, hide them
    if (isENSamplesVisible) {
        for (var i = 0; i < samples.length; i++) {
            samples[i].style.display = 'none';
        }
        document.getElementsByClassName('toggle-ensamples-button')[0].innerHTML = '<i class="fas fa-plus fa-fw"></i>More Samples';
        isENSamplesVisible = false;
    } 
    // If samples are currently hidden, show them
    else {
        for (var i = 0; i < samples.length; i++) {
            samples[i].style.display = 'block';
        }
        document.getElementsByClassName('toggle-ensamples-button')[0].innerHTML = '<i class="fas fa-chevron-up fa-fw"></i>Hide Samples';
        isENSamplesVisible = true;
    }
}
var isENSamplesVisible = false; // Tracks if samples are currently visible

function toggleMIXSamples() {
  // Get all elements with class 'more-mixsamples'
  var samples = document.getElementsByClassName('more-ensamples');

  // If samples are currently visible, hide them
  if (isMIXSamplesVisible) {
    for (var i = 0; i < samples.length; i++) {
        samples[i].style.display = 'none';
    }
    document.getElementsByClassName('toggle-ensamples-button')[0].innerHTML = '<i class="fas fa-plus fa-fw"></i>More Samples';
    isMIXSamplesVisible = false;
} 
// If samples are currently hidden, show them
else {
    for (var i = 0; i < samples.length; i++) {
        samples[i].style.display = 'block';
    }
    document.getElementsByClassName('toggle-ensamples-button')[0].innerHTML = '<i class="fas fa-chevron-up fa-fw"></i>Hide Samples';
    isMIXSamplesVisible = true;
}
}

var isMIXSamplesVisible = false; // Tracks if samples are currently visible

function toggleMIXSamples() {
  // Get all elements with class 'more-mixsamples'
  var samples = document.getElementsByClassName('more-mixsamples');

  // If samples are currently visible, hide them
  if (isMIXSamplesVisible) {
    for (var i = 0; i < samples.length; i++) {
        samples[i].style.display = 'none';
    }
    document.getElementsByClassName('toggle-mixsamples-button')[0].innerHTML = '<i class="fas fa-plus fa-fw"></i>More Samples';
    isMIXSamplesVisible = false;
} 
// If samples are currently hidden, show them
else {
    for (var i = 0; i < samples.length; i++) {
        samples[i].style.display = 'block';
    }
    document.getElementsByClassName('toggle-mixsamples-button')[0].innerHTML = '<i class="fas fa-chevron-up fa-fw"></i>Hide Samples';
    isMIXSamplesVisible = true;
}
}