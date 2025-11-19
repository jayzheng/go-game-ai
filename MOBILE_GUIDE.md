# React Native Mobile App Guide

This guide covers creating mobile apps for iOS and Android using React Native.

## Overview

The mobile app will share the same backend API as the web version, providing:
- Native iOS and Android apps
- Same game features as web
- Offline play capability (future)
- App store distribution

## Prerequisites

- Node.js 16+
- React Native development environment
- For iOS: macOS with Xcode
- For Android: Android Studio

## Setup React Native Environment

### iOS Setup (macOS only)

1. Install Xcode from App Store
2. Install Xcode Command Line Tools:
```bash
xcode-select --install
```

3. Install CocoaPods:
```bash
sudo gem install cocoapods
```

### Android Setup

1. Install [Android Studio](https://developer.android.com/studio)
2. Install Android SDK
3. Configure environment variables in `~/.zshrc` or `~/.bash_profile`:

```bash
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/tools
export PATH=$PATH:$ANDROID_HOME/tools/bin
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

## Create React Native App

### Option 1: Expo (Recommended for Beginners)

Expo provides a simpler development experience:

```bash
cd go-ai-game
npx create-expo-app mobile --template blank
cd mobile
```

Install dependencies:
```bash
npm install axios
npm install react-native-svg  # For drawing game board
```

### Option 2: React Native CLI (More Control)

For more control and native modules:

```bash
cd go-ai-game
npx react-native init mobile
cd mobile
npm install axios
```

## Shared Code Strategy

Since we're using React for web, we can share components with React Native:

### 1. Create Shared Business Logic

```bash
mkdir ../shared
```

Move game logic that can be shared:
- API service calls
- Game state management
- Utility functions

### 2. Platform-Specific Components

Create separate UI components:
- `frontend/src/components/GoBoard.jsx` - Web version (HTML/CSS)
- `mobile/src/components/GoBoard.jsx` - Mobile version (React Native)

## Mobile App Structure

```
mobile/
├── App.js                 # Main app component
├── src/
│   ├── components/
│   │   ├── GoBoard.jsx    # Native game board
│   │   ├── GameInfo.jsx   # Game information panel
│   │   └── GameList.jsx   # Saved games list
│   ├── screens/
│   │   ├── GameScreen.jsx # Main game screen
│   │   └── MenuScreen.jsx # Menu/settings
│   ├── services/
│   │   └── api.js         # API client (can share with web)
│   └── utils/
│       └── helpers.js     # Utility functions
└── package.json
```

## Example Mobile Components

### GoBoard Component (React Native)

```javascript
import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';

const GoBoard = ({ board, onCellClick }) => {
  const boardSize = board.length;
  const cellSize = 40;

  return (
    <Svg width={boardSize * cellSize} height={boardSize * cellSize}>
      {/* Draw grid lines */}
      {/* Draw stones */}
      {/* Handle touches */}
    </Svg>
  );
};
```

### API Service (Can be shared)

The `api.js` from the web app can be reused with minimal changes:

```javascript
// Just update the base URL for mobile
const API_BASE_URL = 'http://YOUR_COMPUTER_IP:8000/api';
```

## Running the Mobile App

### Expo

```bash
cd mobile
npx expo start
```

Then:
- Press `i` for iOS simulator
- Press `a` for Android emulator
- Scan QR code with Expo Go app for physical device

### React Native CLI

iOS:
```bash
npx react-native run-ios
```

Android:
```bash
npx react-native run-android
```

## Connecting to Backend

### Development

Update API URL to your computer's local IP:

```javascript
// mobile/src/services/api.js
const API_BASE_URL = 'http://192.168.1.XXX:8000/api';
```

Find your IP:
- macOS: `ifconfig | grep "inet "`
- Windows: `ipconfig`

### Production

For production, you'll need:
1. Deploy backend to cloud (AWS, Google Cloud, Heroku, etc.)
2. Update API_BASE_URL to production URL
3. Enable HTTPS for security

## App Store Deployment

### iOS App Store

1. **Enroll in Apple Developer Program** ($99/year)
   - https://developer.apple.com/programs/

2. **Configure App**
   - Set bundle identifier
   - Configure app icons and splash screens
   - Set up signing certificates

3. **Build for Production**
   ```bash
   # Expo
   eas build --platform ios

   # React Native CLI
   # Open in Xcode and archive
   ```

4. **Submit to App Store Connect**
   - Upload build
   - Fill app metadata
   - Submit for review

### Google Play Store

1. **Create Google Play Developer Account** ($25 one-time)
   - https://play.google.com/console

2. **Configure App**
   - Set package name
   - Configure app icons and splash screens
   - Generate signing key

3. **Build APK/AAB**
   ```bash
   # Expo
   eas build --platform android

   # React Native CLI
   cd android
   ./gradlew bundleRelease
   ```

4. **Submit to Play Console**
   - Upload APK/AAB
   - Fill store listing
   - Submit for review

## Key Differences: Web vs Mobile

### UI Components

| Web | Mobile |
|-----|--------|
| `<div>` | `<View>` |
| `<span>`, `<p>` | `<Text>` |
| `<button>` | `<TouchableOpacity>` |
| `<input>` | `<TextInput>` |
| CSS | StyleSheet |

### Styling

Web (CSS):
```css
.board {
  display: flex;
  background-color: #dcb35c;
}
```

Mobile (StyleSheet):
```javascript
const styles = StyleSheet.create({
  board: {
    display: 'flex',
    backgroundColor: '#dcb35c'
  }
});
```

### Navigation

Web: React Router
Mobile: React Navigation

```bash
npm install @react-navigation/native
npm install @react-navigation/stack
```

## Testing on Devices

### iOS (Physical Device)
1. Connect iPhone via USB
2. Select device in Xcode
3. Trust computer on device
4. Run app

### Android (Physical Device)
1. Enable Developer Mode
2. Enable USB Debugging
3. Connect via USB
4. Run `adb devices` to verify
5. Run app

## Performance Optimization

1. **Memoize Components**
   ```javascript
   const GoBoard = React.memo(({ board, onCellClick }) => {
     // Component code
   });
   ```

2. **Use FlatList for Lists**
   ```javascript
   <FlatList
     data={games}
     renderItem={renderGameItem}
     keyExtractor={item => item.id}
   />
   ```

3. **Optimize Images**
   - Use appropriate image sizes
   - Consider lazy loading

## Offline Support (Future)

For offline play:

1. **Local Storage**
   ```bash
   npm install @react-native-async-storage/async-storage
   ```

2. **Local AI**
   - Include lightweight AI model
   - Run inference on device

3. **Sync Strategy**
   - Queue moves when offline
   - Sync when back online

## Recommended Libraries

- **Navigation**: `@react-navigation/native`
- **State Management**: `redux` or `zustand`
- **Storage**: `@react-native-async-storage/async-storage`
- **UI Components**: `react-native-paper` or `react-native-elements`
- **Icons**: `react-native-vector-icons`

## Next Steps

1. Set up React Native development environment
2. Create initial mobile app structure
3. Implement core game UI
4. Test on simulators/emulators
5. Test on physical devices
6. Prepare for app store submission
7. Submit to stores

## Resources

- [React Native Docs](https://reactnative.dev/)
- [Expo Docs](https://docs.expo.dev/)
- [iOS Developer](https://developer.apple.com/)
- [Android Developer](https://developer.android.com/)

Good luck with your mobile app!
