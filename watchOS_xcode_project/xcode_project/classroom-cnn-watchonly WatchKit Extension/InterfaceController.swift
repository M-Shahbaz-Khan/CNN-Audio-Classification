//
//  InterfaceController.swift
//  classroom-cnn-watchonly WatchKit Extension
//
//  Created by Shahbaz on 18/05/2020.
//  Copyright © 2020 Shahbaz. All rights reserved.
//

import WatchKit
import Foundation
import UIKit
import Accelerate
import AVFoundation
import CoreGraphics
import CoreML
import WatchConnectivity

class InterfaceController: WKInterfaceController {
    
    let model = classroomClassifier_STFT()

    @IBOutlet weak var lastClassificationLabel: WKInterfaceLabel!
    @IBOutlet weak var lastPrediction: WKInterfaceLabel!
    @IBOutlet weak var recordButtonLabel: WKInterfaceButton!
    @IBOutlet weak var timeRemaining: WKInterfaceLabel!
    @IBOutlet weak var predictionTime: WKInterfaceLabel!
    @IBOutlet weak var lastWinner: WKInterfaceLabel!
    @IBOutlet weak var classification_number: WKInterfaceLabel!
    
    // MARK: - Parameters and Setup
    // Parameters
    let n_fft = 2048 /* Note: change hardcoded value manually in fftSetup variable below to match - that must be a global variable because its initialization is slow, but unfortunately n_fft cannot be used to initialize it since property initializers run before self is available. */
    let hop_length = 512
    let n_mels = 128 // Note changing this will require creating a new mel_basis.json
    let recording_length : Double = 3 // Number of seconds to record, will also require changing model input
    let sampleRate = Double(16000)
    
    var audioEngine = AVAudioEngine()
    let recordingSession = AVAudioSession.sharedInstance()
    let fftSetup = vDSP_create_fftsetup(UInt(ceil(log2(Double(2048)))), Int32(kFFTRadix5))!
    var classificationHistory = Array(repeating: Int(0), count: Int(4))
    var classifications_made = 0
    var recording = false
    
    let device = WKInterfaceDevice.current()
    var currentTimeRemaining = 3
    
    override func awake(withContext context: Any?) {
        super.awake(withContext: context)
        timeRemaining.setText("3s")
        currentTimeRemaining = 3
    }
    
    override func willActivate() {
        // This method is called when watch view controller is about to be visible to user
        //WatchKit.WKExtension.shared().isAutorotating = true // keep the screen awake at certain angles
        device.isBatteryMonitoringEnabled = true
        super.willActivate()
    }
    
    override func didDeactivate() {
        // This method is called when watch view controller is no longer visible
        super.didDeactivate()
    }
    
    @IBAction func startStop() {
        if(recording){
            stopRecording()
            timeRemaining.setText("0:03")
            currentTimeRemaining = 3
            recordButtonLabel.setTitle("Start")
        }else{
            initializeAudioEngine()
            startRecording()
            recordButtonLabel.setTitle("Stop")
        }
    }
    
    func initializeAudioEngine() {
        audioEngine.stop()
        audioEngine.reset()
        
        do {
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            assertionFailure("AVAudioSession setup error : \(error)")
        }
    }
    
    // MARK: - Audio Recording
    func startRecording() {
        print("Started Recording")
        timeRemaining.setText("0:03")
        currentTimeRemaining = 3
        
        let input = audioEngine.inputNode
        let input_format = input.outputFormat(forBus: 0)
        let downsampled_format = AVAudioFormat(commonFormat: AVAudioCommonFormat.pcmFormatFloat32, sampleRate: 16000, channels: 2, interleaved: false)
        let converter = AVAudioConverter(from: input_format, to: downsampled_format!)
        
        recording = true
        /* tempAudioStorage will be used for holding mic buffer data while we wait to collect 3 seconds worth of samples.*/
        var tempAudioStorage = Array(repeating: Float(0), count: (Int(sampleRate * recording_length)))
        var tempAudioIndex = 0
        
        input.installTap(onBus: 0, bufferSize: UInt32(n_fft), format: input_format, block: {(buffer: AVAudioPCMBuffer!, time: AVAudioTime!) -> Void in
            
            var newBufferAvailable = true
            
            let inputCallback: AVAudioConverterInputBlock = { inNumPackets, outStatus in
                if newBufferAvailable {
                    outStatus.pointee = .haveData
                    newBufferAvailable = false
                    return buffer
                } else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
            }
            
            let convertedBuffer = AVAudioPCMBuffer(pcmFormat: downsampled_format!, frameCapacity: AVAudioFrameCount(downsampled_format!.sampleRate) * buffer.frameLength / AVAudioFrameCount(buffer.format.sampleRate))!
            
            var error: NSError?
            let status = converter!.convert(to: convertedBuffer, error: &error, withInputFrom: inputCallback)
            assert(status != .error)

            // Tap will continue to draw PCM data from mic until told to stop.
            // When we get 3s (sampleRate * 3), take the STFT, make predictions, and process will continue to repeat.
            if(tempAudioIndex == Int(self.sampleRate * self.recording_length)){
                // 15 seconds collected
                NSLog("Memory End Record: \(self.memoryFootprint() ?? 0)")
                // Initialize new buffer to put tempAudioStorage back into - probably would be faster to just work with the array, consider rewriting stft function
                var t = clock()
                let oneMinuteBufferFrameCount : AVAudioFrameCount = UInt32(tempAudioIndex)
                let oneMinuteBufferFormat = AVAudioFormat(standardFormatWithSampleRate: self.sampleRate, channels: 2)!
                let oneMinuteBuffer = AVAudioPCMBuffer(pcmFormat: oneMinuteBufferFormat, frameCapacity: oneMinuteBufferFrameCount)
                oneMinuteBuffer?.frameLength = 0
                for i in 0..<tempAudioIndex {
                    oneMinuteBuffer?.floatChannelData!.pointee[i] = tempAudioStorage[i]
                }
                oneMinuteBuffer?.frameLength = AVAudioFrameCount(tempAudioIndex)
                
                // get mel-scaled STFT
                let new_mel_gram = self.stft(buffer: oneMinuteBuffer!)
                self.saveJsonArray2D(numbers: new_mel_gram, name: "mel_gram")
                let stft_model = classroomClassifier_STFT.init()
                
                // Load STFT into data type MLMultiArray required by CoreML
                
                // mel gram coming back with less bins because no zero-padding, fix in stft function
                // Adding zeros to end of mel-gram in STFT function for now
                
                let MLMultArray_specgram = try? MLMultiArray( shape: [1, new_mel_gram[0].count, new_mel_gram.count] as [NSNumber], dataType: .float32)
                let ptr = UnsafeMutablePointer<Float>(OpaquePointer(MLMultArray_specgram?.dataPointer))
                for c in stride(from: (new_mel_gram[0].count - 1), to: 0, by: -1){
                    for r in 0..<new_mel_gram.count{
                        let offset = c * (MLMultArray_specgram?.strides[1].intValue)! +
                            r * (MLMultArray_specgram?.strides[2].intValue)!
                        
                        ptr![offset] = new_mel_gram[r][c]
                    }
                }
                
                // Make Prediction
                let results = try? stft_model.prediction(STFT: MLMultArray_specgram!)
                NSLog("Memory End Classification: \(self.memoryFootprint() ?? 0)")
                //print(results?.Classroom_Activities ?? "Error")

                let activity1 = floor(results!.Classroom_Activities["Class Management"]! * 10000) / 100
                let activity2 = floor(results!.Classroom_Activities["Lecture"]! * 10000) / 100
                let activity3 = floor(results!.Classroom_Activities["Practice"]! * 10000) / 100
                let activity4 = floor(results!.Classroom_Activities["Q&A"]! * 10000) / 100
                
                
                let newPrediction = String(activity1) + ", " + String(activity2) + ", " + String(activity3) + ", " + String(activity4)

                t = clock() - t // Time from receiving audio to delivering prediction
                
                let newPredictionTime = String(floor(Double(t) * 100 / Double(CLOCKS_PER_SEC)) / 100)
                
                // Reset tempAudioIndex to continue capturing from mic
                tempAudioIndex = 0
                
                self.classifications_made += 1
                self.lastPrediction.setText(newPrediction)
                self.predictionTime.setText("Processed in: \(newPredictionTime)s")
                self.lastWinner.setText(results?.Classroom_Activities.max{a, b in a.value < b.value}?.key) // key of max value in results
                let charge_level = self.device.batteryLevel
                
                //print("Prediction CPUTime: \(newPredictionTime) seconds")
                NSLog("\(self.classifications_made) | Prediction CPUTime: \(newPredictionTime) seconds, battery: \(String(charge_level))")
                
                self.classification_number.setText(String(self.classifications_made))
                self.timeRemaining.setText("0:03")
                self.currentTimeRemaining = 3
            }else{
                /* If we don't have one minute of samples yet, keep loading them into tempAudioStorage */
                if(status == AVAudioConverterOutputStatus.haveData){
                    for currentBuffer in 0..<convertedBuffer.frameLength {
                        // 3 seconds not yet collected
                        
                        if(tempAudioIndex < Int(self.sampleRate * self.recording_length)){
                            //tempAudioStorage[tempAudioIndex] = convertedBuffer.floatChannelData!.pointee[Int(currentBuffer)]
                            tempAudioStorage[tempAudioIndex] = Float(convertedBuffer.floatChannelData!.pointee[Int(currentBuffer)])
                            tempAudioIndex += 1
                            
                            // not necessary on watch, remove
                            if(Int(tempAudioIndex) % Int(self.sampleRate) == 0){
                                self.currentTimeRemaining -= 1
                                self.timeRemaining.setText("0:0" + String(self.currentTimeRemaining))
                            }
                        }
                    }
                }
            }
            
            
        })
        
        audioEngine.prepare()
        try! audioEngine.start()
    }
    
    func stopRecording() {
        print("Stopped recording")
        recording = false
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
    }
    
    // MARK: - STFT, FFT and Mel Scaling
    /* Take STFT and apply mel scaling, given 15 seconds of audio PCM data */
    func stft(buffer: AVAudioPCMBuffer) -> [[Float]]{
        let frameCount = UInt32(buffer.frameLength) // Number of valid samples in buffer
        let log2n = UInt(ceil(log2(Double(frameCount))))
        let bufferSizePOT = Int(1 << log2n) // Required because 16-byte aligned realp and imagp below improves performance of vDSP_ctoz
        
        // Initialize all non-valid data to 0
        let leftFrames = buffer.floatChannelData![0]
        
        // create packed real input (alternating successive samples in realp, then imagp)
        var realp = [Float](repeating: 0, count: bufferSizePOT/2)
        var imagp = [Float](repeating: 0, count: bufferSizePOT/2)
        var output = DSPSplitComplex(realp: &realp, imagp: &imagp)
        
        // Load data from buffer into realp and imagp
        leftFrames.withMemoryRebound(to: DSPComplex.self, capacity: bufferSizePOT / 2) {
            vDSP_ctoz($0, 2, &output, 1, UInt(bufferSizePOT / 2))
        }
        
        // Calculate number of frequency bins we will get, considering number of samples, fft size and hop length
        let num_Of_FFT_bins = Int(floor( (Double(frameCount) / Double(hop_length)) - (Double(n_fft) / Double(hop_length)) ))
        var STFT = Array(repeating: Array(repeating: Float(0), count: n_fft/2), count: Int(num_Of_FFT_bins))
        
        var i = 0 // Keeps track of current frequency bin (index for STFT array)
        var currentIndex = Int(0) // Keeps track of beginning of fft window location
        
        // Loop over the number of bins, taking FFTs as we go
        while (i < num_Of_FFT_bins) {
            // slice realp/imagp depending on the bin and get the fft
            let endIndex = Int(currentIndex + (n_fft / 2))
            let currRealpSlice = realp[currentIndex..<endIndex]
            let currRealp: [Float] = Array(currRealpSlice)
            let currImagpSlice = imagp[currentIndex..<endIndex]
            let currImagp: [Float] = Array(currImagpSlice)
            
            STFT[i] = fft(currRealp: currRealp, currImagp: currImagp, n_fft: n_fft)
            
            i += 1
            currentIndex += (hop_length / 2)
        }
        
        // Load mel basis from json array and convert values to floats
        var mel_basis_float = [[Float]](repeating: Array(repeating: Float(0), count: n_fft/4), count: n_mels)
        var j = 0
        let mel_basis = load_mel_basis()
        for array in mel_basis{
            mel_basis_float[j] = array.map {$0.floatValue}
            j += 1
        }
        
        //self.saveJsonArray2D(numbers: STFT, name: "STFT")
        
        /* Flatten STFT and mel basis - required to use the Accelerate matrix operations */
        var OriginalSTFT = STFT.flatMap{$0}
        var mel_basis_flat = Array(repeating: Float(0), count: mel_basis_float.count * mel_basis_float[0].count)
        mel_basis_flat = mel_basis_float.flatMap{$0}
        
        /* Initialize more arrays */
        var mel_spectrogram_flat = Array(repeating: Float(0), count: Int(num_Of_FFT_bins * n_mels))
        var TransposeSTFT = Array(repeating: Float(0), count: Int(num_Of_FFT_bins * (n_fft/4)))
        
        /* Transpose STFT, multiply by mel basis */
        vDSP_mtrans(&OriginalSTFT, 1, &TransposeSTFT, 1, (vDSP_Length(n_fft/4)), vDSP_Length(num_Of_FFT_bins))
        vDSP_mmul(&mel_basis_flat, 1, &TransposeSTFT, 1, &mel_spectrogram_flat, 1, vDSP_Length(n_mels), vDSP_Length(num_Of_FFT_bins), (vDSP_Length(n_fft/4)))
        
        
        var mel_spectrogram = Array(repeating: Array(repeating: Float(1), count: 128), count: num_Of_FFT_bins + 5) // + 5 to compensate for zero-padding issue, fix later
        var m = 0
        for l in (0..<n_mels).reversed(){
            for k in (0..<num_Of_FFT_bins) {
                mel_spectrogram[k][l] = mel_spectrogram_flat[m]
                m += 1
            }
        }
        
        // To compensate for lack of zero-padding FFT, copy values from beginning of spectrogram
        m = 0
        for l in (0..<n_mels).reversed(){
            for k in (num_Of_FFT_bins..<num_Of_FFT_bins + 5) {
                mel_spectrogram[k][l] = mel_spectrogram_flat[m]
                m += 1
            }
        }
        
        // Find max value in array, needed for dB scaling
        let maxValue = Float(mel_spectrogram.map({$0.max()!}).max()!)
        
        // scale appropriately so range is from -100dB to 0dB
        m = 0
        for k in 0..<num_Of_FFT_bins+5 {
            for l in 0..<n_mels {
                var db = (10 * log10(mel_spectrogram[k][l])) - (10 * log10(maxValue))
                
                if(!db.isNormal || db < -100){
                    // In case a log of 0 was taken above, set dB to minimum
                    db = -100
                }else if (db > 0) {
                    db = 0
                }
                
                mel_spectrogram[k][l] = db
                m += 1
            }
        }
        
        NSLog("Memory End mel_gram processing: \(self.memoryFootprint() ?? 0)")
        return mel_spectrogram;
    }
    
    func fft(currRealp: [Float], currImagp: [Float], n_fft: Int) -> [Float]{
        // Copy arguments to make them mutable, create DSPSplitComplex
        var realp = currRealp
        var imagp = currImagp
        
        // Combine to single signal for windowing
        var realpimagp = [Float](repeating: 0, count: Int(n_fft))
        var combined_idx = 0
        for split_idx in 0..<realp.count {
            realpimagp[combined_idx] = realp[split_idx]
            realpimagp[combined_idx + 1] = realp[split_idx]
            
            combined_idx += 2
        }
        
        // Create and apply hann window
        let fft_length = vDSP_Length(n_fft)
        var window = [Float](repeating: 0, count: Int(fft_length))
        let stride = vDSP_Stride(1)
        vDSP_hann_window(&window, fft_length, Int32(vDSP_HANN_DENORM))
        var postWindow = [Float](repeating: 0, count: n_fft)
        vDSP_vmul(&window, stride, realpimagp, stride, &postWindow, stride, fft_length)
        
        combined_idx = 0
        for split_idx in 0..<realp.count {
            realp[split_idx] = postWindow[combined_idx]
            imagp[split_idx] = postWindow[combined_idx + 1]
            combined_idx += 2
        }
        
        var output = DSPSplitComplex(realp: &realp, imagp: &imagp)
        
        // Take FFT
        let log2n = UInt(ceil(log2(Double(n_fft))))
        let bufferSizePOT = Int(1 << log2n)
        vDSP_fft_zrip(fftSetup, &output, 1, vDSP_Length(log2n), Int32(FFT_FORWARD))
        
        // Calculate magnitudes - maybe faster to do this as one step at the end in STFT?
        var fft_mag = [Float](repeating:0.0, count:Int(bufferSizePOT / 4))
        vDSP_zvmags(&output, 1, &fft_mag, 1, vDSP_Length(bufferSizePOT / 4))
        
        return fft_mag
    }
    
    /* Load mel basis JSON array into 2D array */
    func load_mel_basis() -> [[NSNumber]] {
        var mel_basis = [[NSNumber]]()
        
        if let path = Bundle.main.path(forResource: "mel_basis", ofType: "json") {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
                let jsonResult = try JSONSerialization.jsonObject(with: data, options: [])
                if let jsonResult = jsonResult as? [[NSNumber]] {
                    mel_basis = jsonResult
                }
            } catch {
                print("Error loading mel basis.")
            }
        }
        
        return mel_basis
    }

    // MARK: - JSON Saving
    // Test functions to export arrays as json, to load in python (e.g. to plot audio waveform or STFT)
    // Converted to Isuru's answer here to swift 5 https://stackoverflow.com/questions/28768015/how-to-save-an-array-as-a-json-file-in-swift
    func saveJsonArray(numbers: [Float], name: String){
        let documentsDirectoryPathString = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
        let documentsDirectoryPath = NSURL(string: documentsDirectoryPathString)!

        let jsonFilePath = documentsDirectoryPath.appendingPathComponent(name + ".json")
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false

        // creating a .json file in the Documents folder
        if !fileManager.fileExists(atPath: jsonFilePath!.path, isDirectory: &isDirectory) {
            let created = fileManager.createFile(atPath: jsonFilePath!.absoluteString, contents: nil, attributes: nil)
            if created {
                print("File created!")
            } else {
                print("Couldn't create file for some reason")
            }
        } else {
            print("File already exists, overwriting.")
            try? fileManager.removeItem(atPath: jsonFilePath!.absoluteString)
            let created = fileManager.createFile(atPath: jsonFilePath!.absoluteString, contents: nil, attributes: nil)
            if created {
                print("File created!")
            } else {
                print("Couldn't create file for some reason")
            }
            print("File already exists, overwriting.")
        }

        // creating JSON out of the above array
        var jsonData: NSData!
        do {
            jsonData = try JSONSerialization.data(withJSONObject: numbers, options: JSONSerialization.WritingOptions()) as NSData
        } catch let error as NSError {
            print("Array to JSON conversion failed: \(error.localizedDescription)")
        }

        // Write that JSON to the file created earlier
        do {
            let file = try FileHandle(forWritingTo: jsonFilePath!)
            file.write(jsonData as Data)
            print("JSON data was written to the file successfully!")
        } catch let error as NSError {
            print("Couldn't write to file: \(error.localizedDescription)")
        }
    }
    
    func saveJsonArray2D(numbers: [[Float]], name: String){
        let documentsDirectoryPathString = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
        let documentsDirectoryPath = NSURL(string: documentsDirectoryPathString)!

        let jsonFilePath = documentsDirectoryPath.appendingPathComponent(name + ".json")
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false

        // creating a .json file in the Documents folder
        if !fileManager.fileExists(atPath: jsonFilePath!.path, isDirectory: &isDirectory) {
            let created = fileManager.createFile(atPath: jsonFilePath!.absoluteString, contents: nil, attributes: nil)
            if created {
                print("File created!")
            } else {
                print("Couldn't create file for some reason")
            }
        } else {
            print("File already exists, overwriting.")
            let created = fileManager.createFile(atPath: jsonFilePath!.absoluteString, contents: nil, attributes: nil)
            if created {
                print("File created!")
            } else {
                print("Couldn't create file for some reason")
            }
        }

        // creating JSON out of the above array
        var jsonData: NSData!
        do {
            jsonData = try JSONSerialization.data(withJSONObject: numbers, options: JSONSerialization.WritingOptions()) as NSData
        } catch let error as NSError {
            print("Array to JSON conversion failed: \(error.localizedDescription)")
        }

        // Write that JSON to the file created earlier
        do {
            let file = try FileHandle(forWritingTo: jsonFilePath!)
            file.write(jsonData as Data)
            print("JSON data was written to the file successfully!")
        } catch let error as NSError {
            print("Couldn't write to file: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Memory Monitoring
    func memoryFootprint() -> mach_vm_size_t? {
        // The `TASK_VM_INFO_COUNT` and `TASK_VM_INFO_REV1_COUNT` macros are too
        // complex for the Swift C importer, so we have to define them ourselves.
        let TASK_VM_INFO_COUNT = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let TASK_VM_INFO_REV1_COUNT = mach_msg_type_number_t(MemoryLayout.offset(of: \task_vm_info_data_t.min_address)! / MemoryLayout<integer_t>.size)
        var info = task_vm_info_data_t()
        var count = TASK_VM_INFO_COUNT
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard
            kr == KERN_SUCCESS,
            count >= TASK_VM_INFO_REV1_COUNT
        else { return nil }
        return info.phys_footprint
    }
    
    // End of class
}
