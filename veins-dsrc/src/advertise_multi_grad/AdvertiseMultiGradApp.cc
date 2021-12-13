//
// Copyright (C) 2016 David Eckhoff <david.eckhoff@fau.de>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "advertise_multi_grad/AdvertiseMultiGradApp.h"

#include <string>

#include "msgs/RequestGradientMessage_m.h"
#include "msgs/SendGradientMessage_m.h"

using namespace veins;

Define_Module(veins::AdvertiseMultiGradApp);

static const std::string gradient_vehicle_color = "blue";
static const uint32_t MAX_WAIT_TIME = 2;

void AdvertiseMultiGradApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Initializing members and pointers of your application goes here
        EV << "Initializing " << par("appName").stringValue() << std::endl;
        broadcasting = false;
        transfers_in_progress = 0;
        prev_transfers_in_progress = 0;
    }
    else if (stage == 1) {
        // Initializing members that require initialized other modules goes here
        std::string node_name = findHost()->getFullName();
        if (node_name == "node[37]") {
          gradientHash = myId + 1;
          findHost()->getDisplayString().setTagArg("i", 1, gradient_vehicle_color.c_str());
          gradientCount = 1;
        } else {
          gradientHash = 0;
          gradientCount = 0;
        }
        EV << "Initialized " << node_name << " with vehicle ID " << myId
           << " and gradientHash " << gradientHash << std::endl;
    }
}

void AdvertiseMultiGradApp::finish()
{
    DemoBaseApplLayer::finish();
    // statistics recording goes here
    if (gradientCount < 1) {
        gradientCount = 0;
    }
    recordScalar("gradientCount", gradientCount);
}

void AdvertiseMultiGradApp::onBSM(DemoSafetyMessage* bsm)
{
    // Your application has received a beacon message from another car or RSU
    // code for handling the message goes here
}

void AdvertiseMultiGradApp::onWSM(BaseFrame1609_4* wsm)
{
    // Your application has received a data message from another car or RSU
    // code for handling the message goes here, see TraciDemo11p.cc for examples
    if (RequestGradientMessage* rgm = dynamic_cast<RequestGradientMessage*>(wsm)) {
        LAddress::L2Type senderAddress = rgm->getSenderAddress();
        EV << findHost()->getFullName() << " received request from " << senderAddress << std::endl;
        
        SendGradientMessage* wsm = new SendGradientMessage("gradients");
        populateWSM(wsm, senderAddress);
        wsm->setSenderAddress(myId);
        wsm->setGradientHash(gradientHash);
        EV << findHost()->getFullName() << " sending gradients to " << senderAddress << std::endl;
        scheduleAt(simTime(), wsm);
        transfers_in_progress = transfers_in_progress + 2;
    } else if (SendGradientMessage* sgm = dynamic_cast<SendGradientMessage*>(wsm)) {        
        LAddress::L2Type senderAddress = sgm->getSenderAddress();
        int senderGradientHash = sgm->getGradientHash();
        if (receivedAddresses.find(senderAddress) != receivedAddresses.end()) {
            gradientCount = 1;
            gradientHash = senderGradientHash;
            // Set vehicle to blue
            findHost()->getDisplayString().setTagArg("i", 1, gradient_vehicle_color.c_str());
        } else {
            EV << findHost()->getFullName() << " received first gradientHash " \
               << senderGradientHash << " from " << senderAddress << std::endl;
            receivedAddresses.insert(senderAddress);
            gradientCount = std::max(gradientCount, 0.5);
        }
    }
}

void AdvertiseMultiGradApp::onWSA(DemoServiceAdvertisment* wsa)
{
    // Your application has received a service advertisement from another car or RSU
    // code for handling the message goes here, see TraciDemo11p.cc for examples
    LAddress::L2Type senderAddress = wsa->getPsid();
    EV << findHost()->getFullName() << " received WSA from senderAddress " << senderAddress << std::endl;
    
    // Request gradients if sender not in receivedAddresses
    if (receivedAddresses.find(senderAddress) == receivedAddresses.end()) {
        RequestGradientMessage* wsm = new RequestGradientMessage("request");
        populateWSM(wsm, senderAddress);
        wsm->setSenderAddress(myId);
        EV << findHost()->getFullName() << " requesting gradients from " << senderAddress << std::endl;
        sendDown(wsm);
    } else {
        EV << findHost()->getFullName() << " already received gradients from " << senderAddress << std::endl;
    }
}

void AdvertiseMultiGradApp::handleSelfMsg(cMessage* msg)
{
    if (transfers_in_progress == 0) {
        DemoBaseApplLayer::handleSelfMsg(msg);
    }
    if (SendGradientMessage* sgm = dynamic_cast<SendGradientMessage*>(msg)) {
        sendDown(sgm);
        transfers_in_progress = transfers_in_progress - 1;
        if (transfers_in_progress > 0) {
            scheduleAt(simTime() + MAX_WAIT_TIME - 1, sgm->dup());
        }
    }
    // this method is for self messages (mostly timers)
    // it is important to call the DemoBaseApplLayer function for BSM and WSM transmission
}

void AdvertiseMultiGradApp::handlePositionUpdate(cObject* obj)
{
    // the vehicle has moved. Code that reacts to new positions goes here.
    // member variables such as currentPosition and currentSpeed are updated in the parent class
    DemoBaseApplLayer::handlePositionUpdate(obj);

    if (transfers_in_progress == prev_transfers_in_progress) {
        transfers_in_progress = 0;
    }
    prev_transfers_in_progress = transfers_in_progress;

    // If vehicle has gradients
    if (gradientHash != 0 && ! broadcasting) {
        startService(Channel::sch2, myId, "Gradients");
        EV << findHost()->getFullName() << " has started broadcasting" << std::endl;
        broadcasting = true;
    }

    if (gradientCount < 1) {
        gradientCountVector.recordWithTimestamp(simTime(), 0);
    } else {
        gradientCountVector.recordWithTimestamp(simTime(), gradientCount);
    }
}
